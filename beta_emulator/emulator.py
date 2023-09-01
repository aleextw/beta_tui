# pylint: disable=invalid-name, missing-module-docstring, missing-function-docstring, eval-used
import logging
import math
import re
import inspect
import logging

from collections import defaultdict
from copy import deepcopy
from deepdiff import DeepDiff
from functools import wraps
from pathlib import Path
from typing import Optional, Union
from bitstring import BitArray
from beta_emulator.memory import Memory
from beta_emulator.registers import Register

func_mapper = defaultdict(dict)


def methoddispatch(func):
    global func_mapper

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func_mapper[func.__name__][len(args)](*args, **kwargs)

    def register(overload_func):
        func_mapper[func.__name__][
            len(inspect.signature(overload_func).parameters)
        ] = overload_func
        return overload_func

    func_mapper[func.__name__][len(inspect.signature(func).parameters)] = func
    wrapper.register = register
    return wrapper


class Emulator:
    """
    Beta emulator
    """

    ### REGISTER CONSTANTS ###
    # Number of registers
    NUM_REGS = 32

    # (Zero-indexed) index of special registers
    ZP = 31
    XP = 30
    SP = 29
    LP = 28
    BP = 27

    # Width of each register in bytes
    # Default is 4 (32-bit registers)
    REG_WIDTH = 4

    #########################

    ### MEMORY CONSTANTS ####
    # If modifying these constants, ensure that it doesn't
    # exceed the memory capacity available on the Alchitry AU
    # or whatever FPGA you are using

    # Size of instruction memory in bytes
    # Leave as default if unsure
    # INSTR_MEM_SIZE = 1 * 1024 * 1024
    INSTR_MEM_SIZE = 1 * 1024

    # Size of data memory in bytes
    # Leave as default if unsure
    # DATA_MEM_SIZE = 1 * 1024 * 1024
    DATA_MEM_SIZE = 4 * 1024

    # Width of each instruction in bytes
    # Default is 4 (32-bit instructions)
    INSTR_WIDTH = 4

    # Width of each opcode in bits
    # Ensure that opcode width + constant + log(NUM_REGS) <= INSTR_MEM_SIZE
    # Default is 6
    OPCODE_WIDTH = 6

    # Width of literal in bits
    # Ensure that opcode width + constant + log(NUM_REGS) <= INSTR_MEM_SIZE
    # Default is 6
    LITERAL_WIDTH = 16

    # Width of each memory word in bytes
    # Default is 4 (32-bit memory)
    DATA_WIDTH = 4

    #########################

    def __init__(self):
        # Number of bits used to index registers
        self.REG_NUM_BITS = math.ceil(math.log2(self.NUM_REGS))

        # Number of unused bits in an OP-type instruction
        self.OP_UNUSED_SIZE = (
            self.INSTR_WIDTH * 8 - (3 * self.REG_NUM_BITS) - self.OPCODE_WIDTH
        )

        # Number of unused bits in an OPC-type instruction
        self.OPC_UNUSED_SIZE = (
            self.INSTR_WIDTH * 8
            - (2 * self.REG_NUM_BITS)
            - self.LITERAL_WIDTH
            - self.OPCODE_WIDTH
        )

        # Check if config settings respect OP-type instruction limits
        if self.OP_UNUSED_SIZE < 0:
            error_message = "Emulator failed to initialize: OP_UNUSED_SIZE < 0"
            logging.error(error_message)
            raise ValueError(error_message)

        # Check if config settings respect OPC-type instruction limits
        if self.OPC_UNUSED_SIZE < 0:
            error_message = "Emulator failed to initialize: OPC_UNUSED_SIZE < 0"
            logging.error(error_message)
            raise ValueError(error_message)

        # Initialize bytes used for instruction memory and data memory
        self.instruction_memory = Memory(
            self.INSTR_MEM_SIZE, self.INSTR_WIDTH, "INSTRUCTION MEMORY"
        )
        self.data_memory = Memory(self.DATA_MEM_SIZE, self.DATA_WIDTH, "DATA MEMORY")

        self.registers = [
            Register(self.REG_WIDTH, idx == self.ZP, f"R{idx}")
            for idx in range(self.NUM_REGS)
        ]

        self.program_counter = 0

        # Assembler variables
        self.data_memory_backup = None
        self.parser_memory_reference = None
        self.partial_result = BitArray()
        self.asm_variables = {}

        # Debugging variables
        self.history = []
        self.frame = {}

        """
        The format of each frame in the history stack is as follows:
        {
            "data_memory": {
                "address": "",
                "value": ""
            },
            "registers": {
                "register_number": "",
                "value": ""
            },
            "program_counter": ""
        }

        Note that not all entries will be present, e.g.,
        if we run ADD(), only registers and the program counter will be affected,
        hence there won't be a data_memory entry

        Additionally, you should do some checking before pushing a frame
        onto the history stack to see if the topmost frame is identical
        - If it is (e.g., in the case of a HALT() loop), then don't append the frame,
          otherwise, you'll end up having to pop off a million duplicate frames 
          while backtracking
        """

    ### EMULATION FUNCTIONS ###

    def load_files(
        self,
        instruction_memory_filepath: Path,
        data_memory_filepath: Optional[Path] = None,
    ):
        self.parser_memory_reference = self.data_memory
        if data_memory_filepath:
            with open(data_memory_filepath, encoding="utf8") as file:
                data_memory = file.read()
            self.parse_data_memory(data_memory)

        # Make a copy of the data memory in case we reset
        self.data_memory_backup = deepcopy(self.data_memory.data)

        self.parser_memory_reference = self.instruction_memory

        with open(instruction_memory_filepath, encoding="utf8") as file:
            instruction_memory = file.read()
        self.parse_instruction_memory(instruction_memory)
        self.reset()

        # Write instruction memory to memory unit
        for idx, val in enumerate(self.instruction_memory.unparsed_data):
            if val is None:
                continue

            command, *params = val
            command_output = type(self).__dict__[command](
                self,
                *[
                    self.parse_param(i.strip()) if not isinstance(i, int) else i
                    for i in params
                ],
            )
            self.instruction_memory.set(idx * self.INSTR_WIDTH, command_output)

        self.reset()

    def reset(self):
        self.program_counter = 0
        for register in self.registers:
            register.reset()

        # Write data memory to memory unit
        # Only need to reset data memory since instruction
        # memory isn't allowed to be changed
        for idx, val in enumerate(self.data_memory_backup):
            self.data_memory.set(idx * self.DATA_WIDTH, val.uint)

    def step_forward(self):
        if (
            instruction := self.instruction_memory.get_instr(self.program_counter)
        ) is None:
            # We loop here instead of just exiting the loop so that we can
            # step through execution even on a HALT() instruction
            return

        command, *params = instruction

        type(self).__dict__[command](
            self,
            *[
                self.parse_param(i.strip()) if not isinstance(i, int) else i
                for i in params
            ],
        )

        if not self.history or DeepDiff(self.history[-1], self.frame):
            self.history.append(deepcopy(self.frame))
            self.frame = {}

    def step_backward(self):
        if not self.history:
            return

        frame = self.history.pop()

        if "data_memory" in frame:
            self.data_memory.set(
                frame["data_memory"]["address"], frame["data_memory"]["value"]
            )

        if "registers" in frame:
            self.registers[frame["registers"]["register_number"]].set(
                frame["registers"]["value"]
            )

        if "program_counter" in frame:
            self.program_counter = frame["program_counter"]

    # The asterisk enforces that all arguments to this method
    # have to be passed via keyword only
    def write_debug_frame(
        self,
        *,
        register_idx: Optional[int] = None,
        data_memory_address: Optional[int] = None,
    ):
        """
        Used to generate frame with change information for the
        emulator to use when stepping backward through execution

        register_idx: Optional[int] - The index of the register that
            will be modified after execution of this instruction

        data_memory_address: Optional[int] - The address in the data
            memory that will be modified afer execution of this instruction
        """
        if register_idx is not None:
            self.frame["registers"] = {
                "register_number": register_idx,
                "value": self.registers[register_idx].get(),
            }

        if data_memory_address is not None:
            self.frame["data_memory"] = {
                "address": data_memory_address,
                "value": self.data_memory.get(data_memory_address),
            }

        self.frame["program_counter"] = self.program_counter

    ###########################

    ### OP INSTRUCTIONS ###
    def ADD(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() + self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("100000", ra, rb, rc)

    def SUB(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() - self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("100001", ra, rb, rc)

    def MUL(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() * self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("100010", ra, rb, rc)

    def DIV(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() // self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("100011", ra, rb, rc)

    def AND(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() & self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("101000", ra, rb, rc)

    def OR(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() | self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("101001", ra, rb, rc)

    def XOR(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() ^ self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("101010", ra, rb, rc)

    def CMPEQ(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() == self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("100100", ra, rb, rc)

    def CMPLT(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() < self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("100101", ra, rb, rc)

    def CMPLE(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() <= self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("100110", ra, rb, rc)

    def SHL(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() << self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("101100", ra, rb, rc)

    def SHR(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(
            (
                BitArray(int=self.registers[ra].get(), length=self.REG_WIDTH * 8)
                >> self.registers[rb].get()
            ).int
        )
        self.program_counter += 4
        return self._build_op_output("101101", ra, rb, rc)

    def SRA(self, ra: int, rb: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() >> self.registers[rb].get())
        self.program_counter += 4
        return self._build_op_output("101110", ra, rb, rc)

    #######################

    ### OPC INSTRUCTIONS ###
    def ADDC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() + literal)
        self.program_counter += 4
        return self._build_opc_output("110000", ra, literal, rc)

    def SUBC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() - literal)
        self.program_counter += 4
        return self._build_opc_output("110001", ra, literal, rc)

    def MULC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() * literal)
        self.program_counter += 4
        return self._build_opc_output("110010", ra, literal, rc)

    def DIVC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() // literal)
        self.program_counter += 4
        return self._build_opc_output("110011", ra, literal, rc)

    def ANDC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() & literal)
        self.program_counter += 4
        return self._build_opc_output("111000", ra, literal, rc)

    def ORC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() | literal)
        self.program_counter += 4
        return self._build_opc_output("111001", ra, literal, rc)

    def XORC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() ^ literal)
        self.program_counter += 4
        return self._build_opc_output("111010", ra, literal, rc)

    def CMPEQC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() == literal)
        self.program_counter += 4
        return self._build_opc_output("110100", ra, literal, rc)

    def CMPLTC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() < literal)
        self.program_counter += 4
        return self._build_opc_output("110101", ra, literal, rc)

    def CMPLEC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() <= literal)
        self.program_counter += 4
        return self._build_opc_output("110110", ra, literal, rc)

    def SHLC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() << literal)
        self.program_counter += 4
        return self._build_opc_output("111100", ra, literal, rc)

    def SHRC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(
            (
                BitArray(int=self.registers[ra].get(), length=self.REG_WIDTH * 8)
                >> literal
            ).int
        )
        self.program_counter += 4
        return self._build_opc_output("111101", ra, literal, rc)

    def SRAC(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.registers[ra].get() >> literal)
        self.program_counter += 4
        return self._build_opc_output("111110", ra, literal, rc)

    ########################

    ### Other INSTRUCTIONS ###
    @methoddispatch
    def LD(self, ra: int, literal: int, rc: int):
        self.write_debug_frame(
            register_idx=rc, data_memory_address=self.registers[ra].get()
        )
        self.registers[rc].set(self.data_memory.get(self.registers[ra].get() + literal))

        self.program_counter += 4
        return self._build_opc_output("011000", ra, literal, rc)

    @methoddispatch
    def ST(self, rc: int, literal: int, ra: int):
        self.write_debug_frame(data_memory_address=self.registers[ra].get() + literal)
        self.data_memory.set(
            self.registers[ra].get() + literal, self.registers[rc].get()
        )
        self.program_counter += 4
        return self._build_opc_output("011001", ra, literal, rc)

    @methoddispatch
    def JMP(self, ra: int, rc: int):
        self.write_debug_frame(register_idx=rc)
        self.registers[rc].set(self.program_counter + 4)

        self.program_counter = self.mask_highest_address_bit(self.registers[ra].get())
        return self._build_opc_output("011011", ra, 0, rc)

    @methoddispatch
    def BEQ(self, ra: int, label: str, rc: int):
        self.write_debug_frame(register_idx=rc)
        literal = (label - (self.program_counter + 4)) // self.INSTR_WIDTH

        self.registers[rc].set(self.program_counter + 4)

        if not self.registers[ra].get():
            self.program_counter = self.program_counter + 4 + 4 * literal
        else:
            self.program_counter += 4
        return self._build_opc_output("011101", ra, literal, rc)

    @methoddispatch
    def BF(self, ra: int, label: str, rc: int):
        return self.BEQ(ra, label, rc)

    @methoddispatch
    def BNE(self, ra: int, label: str, rc: int):
        self.write_debug_frame(register_idx=rc)
        literal = (label - (self.program_counter + 4)) // self.INSTR_WIDTH

        self.registers[rc].set(self.program_counter + 4)

        if self.registers[ra].get():
            self.program_counter = self.program_counter + 4 + 4 * literal
        else:
            self.program_counter += 4
        return self._build_opc_output("011110", ra, literal, rc)

    @methoddispatch
    def BT(self, ra: int, label: str, rc: int):
        return self.BNE(ra, label, rc)

    def LDR(self, label: str, rc: int):
        self.write_debug_frame(register_idx=rc)
        literal = (label - (self.program_counter + 4)) // self.INSTR_WIDTH

        self.registers[rc].set(
            self.data_memory.get(self.program_counter + 4 + 4 * literal)
        )
        self.program_counter += 4
        return self._build_opc_output("011111", 0, literal, rc)

    def HALT(self):
        return self._build_opc_output("000000", 0, 0, 0)

    ##########################

    ### MACRO INSTRUCTIONS ###

    # Since data can be written in bytes only,
    # the WORD and LONG macros are used to assemble
    # 2-byte and 4-byte values respectively
    #
    # Therefore, if doing 16-bit machines, you
    # will only need to use WORD
    def WORD(self, data: int):
        return BitArray(int=data, length=16).bin

    def LONG(self, data: int):
        return BitArray(int=data, length=32).bin

    # If you put a @methoddispatch decorator
    # over another function, you can overload it
    # by decorating another function with @funcName.register
    # to get some semblance of overloading in Python
    @BEQ.register
    def _(self, ra: int, label: str):
        return self.BEQ(ra, label, 31)

    @BF.register
    def _(self, ra: int, label: str):
        return self.BF(ra, label, 31)

    @BNE.register
    def _(self, ra: int, label: str):
        return self.BNE(ra, label, 31)

    @BT.register
    def _(self, ra: int, label: str):
        return self.BT(ra, label, 31)

    @BEQ.register
    @methoddispatch
    def BR(self, label: str, rc: int):
        return self.BEQ(31, label, rc)

    @BR.register
    def _(self, label: str):
        return self.BR(label, 31)

    @JMP.register
    def _(self, ra: int):
        return self.JMP(ra, 31)

    @LD.register
    def _(self, label: str, rc: int):
        return self.LD(31, label, rc)

    @ST.register
    def _(self, rc: int, label: str):
        return self.ST(rc, label, 31)

    def MOVE(self, ra: int, rc: int):
        return self.ADD(ra, 31, rc)

    def CMOVE(self, const: int, rc: int):
        return self.ADDC(31, const, rc)

    def PUSH(self, ra: int):
        return [["ADDC", "SP", "4", "SP"], ["ST", ra, "-4", "SP"]]

    def POP(self, rc: int):
        return [["LD", "SP", "-4", rc], ["SUBC", "SP", "4", "SP"]]

    def ALLOCATE(self, k: int):
        return self.ADDC(self.SP, 4 * k, self.SP)

    def DEALLOCATE(self, k: int):
        return self.SUBC(self.SP, 4 * k, self.SP)

    ##########################

    ### HELPER FUNCTIONS ###

    def mask_highest_address_bit(self, address: int):
        """
        Mask the program counter with instruction address bit 31
        to prevent undesired kernel entry
        """
        return address & int(
            str(self.program_counter >> (self.INSTR_WIDTH * 4 - 1))
            + "1" * (self.INSTR_WIDTH * 4 - 1),
            base=2,
        )

    def _build_op_output(self, opcode: str, ra: int, rb: int, rc: int) -> int:
        base = (
            BitArray(bin=opcode, length=self.OPCODE_WIDTH)
            + BitArray(uint=rc, length=self.REG_NUM_BITS)
            + BitArray(uint=ra, length=self.REG_NUM_BITS)
            + BitArray(uint=rb, length=self.REG_NUM_BITS)
        )

        if self.OP_UNUSED_SIZE:
            base += BitArray(int=0, length=self.OP_UNUSED_SIZE)
        return base.uint

    def _build_opc_output(self, opcode: str, ra: int, literal: int, rc: int) -> int:
        base = (
            BitArray(bin=opcode, length=self.OPCODE_WIDTH)
            + BitArray(uint=rc, length=self.REG_NUM_BITS)
            + BitArray(uint=ra, length=self.REG_NUM_BITS)
            + BitArray(int=literal, length=self.LITERAL_WIDTH)
        )

        if self.OPC_UNUSED_SIZE:
            base += BitArray(int=0, length=self.OPC_UNUSED_SIZE)
        return base.uint

    ########################

    ### ASSEMBLER FUNCTIONS ###

    def parse_instruction_memory(self, data: str):
        self.asm_variables["."] = 0
        data = re.sub(r"\| *.*(\n|)", "", data).replace("\n", " ").strip()
        # First pass, capture all labels
        while data:
            if match := self.is_label(data):
                self.asm_variables[
                    data[match.start() : match.end()].replace(":", "").strip()
                ] = self.asm_variables["."]
            elif match := self.is_command(data):
                # We try to eval all commands here
                # then check if the return value is a list
                # If it is a list, then most likely a macro,
                # we should replace the current instruction with the list
                args = self.attempt_parse_command_params(
                    data[match.start() : match.end()].strip()
                )
                try:
                    command, *params = args
                    command_output = type(self).__dict__[command](self, *params)
                    if isinstance(command_output, list):
                        args = command_output
                    else:
                        args = [args]
                except Exception:
                    args = [args]
                for arg in args:
                    self.instruction_memory.set_instr(
                        self.asm_variables["."],
                        arg,
                    )
                    self.asm_variables["."] += self.INSTR_WIDTH
            elif match := self.is_equation(data):
                self.parse_equation(data[match.start() : match.end()].strip())
            else:
                error_message = f"Failed to parse token: {data[:10]}..."
                logging.error(error_message)
                raise ValueError(error_message)
            data = data[match.end() :].strip()
        # Second pass, all labels should have been captured, and any variables defined
        # when evaluating instruction parameters should have been replaced
        # Instructions should also be stored as a list in the following format:
        # [INSTRUCTION, [PARAM1, PARAM2, ...]]
        for instruction in self.instruction_memory.unparsed_data:
            if instruction is None or len(instruction) == 1:
                continue

            for idx, val in enumerate(instruction[1:], start=1):
                if not isinstance(val, int):
                    instruction[idx] = self.parse_param(val)

    def attempt_parse_command_params(self, text: str):
        # For now we assume that data memory commands will only be WORD or LONG
        command, *params = text.replace(")", "").replace("(", ",").split(",")

        if command not in type(self).__dict__:
            error_message = f"Couldn't find an instruction with name {command}"
            logging.error(error_message)
            raise ValueError(error_message)

        # We attempt to parse the instruction parameters in case there is some self-assignment
        # to a variable further down the code, which would render the parse inaccurate if we
        # finished evaluating variables on the first pass, then only assigned its value to a
        # parameter using it on the second pass
        params = [i for i in params if i != ""]
        for idx, param in enumerate(params):
            param = param.strip()
            try:
                params[idx] = self.parse_param(param)
            except ValueError as _:
                logging.info(
                    "Failed to parse token %s on first pass, delaying evaluation.",
                    param,
                )
                # Still write back even though failed since we removed whitespace
                # which will make second-pass processing easier
                params[idx] = param

        return [command, *params]

    def parse_data_memory(self, data: str):
        self.asm_variables["."] = 0

        data = re.sub(r"\| *.*(\n|)", "", data).replace("\n", " ")

        while data:
            if match := self.is_label(data):
                self.asm_variables[
                    data[match.start() : match.end()].replace(":", "").strip()
                ] = self.asm_variables["."]
            elif match := self.is_command(data):
                self.append_command_to_result(data[match.start() : match.end()].strip())
            elif match := self.is_equation(data):
                self.parse_equation(data[match.start() : match.end()].strip())
            elif match := self.is_expr(data):
                self.append_data_to_result(
                    self.parse_expr(data[match.start() : match.end()].strip())
                )
            elif match := self.is_variable(data):
                self.append_data_to_result(
                    self.asm_variables[data[match.start() : match.end()].strip()]
                )
            elif match := self.is_numeric(data):
                self.append_data_to_result(data[match.start() : match.end()].strip())
            else:
                error_message = f"Failed to parse token: {data[:10]}..."
                logging.error(error_message)
                raise ValueError(error_message)
            data = data[match.end() :].strip()

        if self.partial_result:
            logging.warning(
                "Remaining data in partial result, data might be missing or misaligned"
            )

    def append_data_to_result(self, text: Union[str, int]):
        if isinstance(text, str):
            data = BitArray(int=self.parse_numeral(text) & 255, length=8)
        else:
            data = BitArray(int=text & 255, length=8)

        self.partial_result += data
        if self.partial_result.length >= self.DATA_WIDTH * 8:
            self.add_partial_result()

    def append_command_to_result(self, text: str):
        # For now we assume that data memory commands will only be WORD or LONG
        command, *params = text.replace(")", "").replace("(", ",").split(",")

        if command not in type(self).__dict__:
            error_message = f"Couldn't find an instruction with name {command}"
            logging.error(error_message)
            raise ValueError(error_message)

        # Do this way rather than just doing Emulator.__dict__
        # in case someone tries to inherit from Emulator rather
        # than editing emulator directly
        command_output = type(self).__dict__[command](
            self, *[self.parse_param(i.strip()) for i in params]
        )
        self.partial_result.bin += command_output

        # Ideally, the length would be == self.DATA_WIDTH
        # but idk what kind of ASM code people are gonna be writing so
        if self.partial_result.length >= self.DATA_WIDTH * 8:
            self.add_partial_result()

    def add_partial_result(self):
        self.data_memory.set(
            self.asm_variables["."], self.partial_result[: self.DATA_WIDTH * 8].int
        )
        # self.data_memory_results.append(self.partial_result[: self.DATA_WIDTH * 8])
        self.partial_result = self.partial_result[self.DATA_WIDTH * 8 :]
        self.asm_variables["."] += self.DATA_WIDTH

    def parse_param(self, data: str):
        # TODO: Fix this shitty regex so that it matches the whole string and I don't have
        #       to bodge it like this
        if (
            (match := self.is_expr(data))
            and match.start() == 0
            and match.end() == len(data)
        ):
            return self.parse_expr(data)
        if data in self.asm_variables:
            return self.asm_variables[data]
        if (
            (match := self.is_numeric(data))
            and match.start() == 0
            and match.end() == len(data)
        ):
            return self.parse_numeral(data)
        if data and data[0] == "R":
            return self.parse_numeral(data[1:])
        if data and data in Emulator.__dict__:
            return Emulator.__dict__[data]

        error_message = f"Unidentified token encountered: {data}"
        logging.error(error_message)
        raise ValueError(error_message)

    def parse_expr(self, data: str):
        # Using eval() since it's gonna run locally
        # so if anyone's machine screws up, it's gonna be yours
        return eval(data, deepcopy(self.asm_variables))

    def parse_equation(self, data: str):
        variable_name, expression = data.split("=")
        self.asm_variables[variable_name.strip()] = self.parse_expr(expression)

    def parse_numeral(self, data: str):
        if data[:2] == "0b":
            return int(data, base=2)
        if data[:2] == "0x":
            return int(data, base=16)
        if self.is_numeric(data):
            return int(data)
        error_message = f"Failed to parse token {data}"
        logging.error(error_message)
        raise ValueError(error_message)

    def is_command(self, text: str) -> re.Match:
        return re.search(r"^ ?\S+ *\((( ?\S+, ?)*\S+|) ?\)", text)

    def is_numeric(self, text: str) -> re.Match:
        return re.search(r"^ ?(0[bB][01]+)|(0[xX][0-9a-zA-Z]+)|(-?[0-9]+)", text)

    def is_variable(self, text: str) -> re.Match:
        return re.search("^(" + "|".join(self.asm_variables.keys()) + ")", text)

    def is_equation(self, text: str) -> re.Match:
        return re.search(
            r"^ ?[a-zA-Z_]+ ?= ?(-?([0-9]|[a-zA-Z_])+ ?(\+|\-|\*|\/|\%|<<|>>) ?)*-?([0-9]|[a-zA-Z_])+ ?",
            text,
        )

    def is_expr(self, text: str) -> re.Match:
        return re.search(r"^ ?(-?[0-9]+ ?(\+|\-|\*|\/|\%|<<|>>) ?)+-?[0-9]+ ?", text)

    def is_label(self, text: str) -> re.Match:
        return re.search(r"^ ?[a-zA-Z][a-zA-Z0-9_]* *:", text)

    ###########################


if __name__ == "__main__":
    emulator = Emulator()

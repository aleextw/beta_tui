# pylint: disable=missing-function-docstring, missing-module-docstring, redefined-outer-name
import pytest
from bitstring import BitArray

from beta_emulator.emulator import Emulator
from beta_emulator.registers import Register


@pytest.fixture
def writeable_register():
    return Register(32, False, "R1")


@pytest.fixture
def readonly_register():
    return Register(32, True, "R1")


@pytest.fixture
def emulator():
    return Emulator()


def build_bin_string(literal: int):
    return BitArray(int=literal, length=16).bin


def test_register_get(writeable_register: Register):
    assert writeable_register.get() == 0


def test_register_set(writeable_register: Register):
    writeable_register.set(5)
    assert writeable_register.get() == 5


def test_register_reset(writeable_register: Register):
    writeable_register.set(10)
    assert writeable_register.get() == 10
    writeable_register.reset()
    assert writeable_register.get() == 0


def test_register_readonly(readonly_register: Register):
    readonly_register.set(10)
    assert readonly_register.get() == 0


def test_add_no_overflow(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(10)
    assert emulator.ADD(0, 1, 2) == int(
        "100000" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 15


def test_add_overflow(emulator: Emulator):
    emulator.registers[0].set(2**31 - 1)
    emulator.registers[1].set(1)
    assert emulator.ADD(0, 1, 2) == int(
        "100000" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == -(2**31)


def test_sub_no_overflow(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(10)
    assert emulator.SUB(0, 1, 2) == int(
        "100001" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == -5


def test_sub_overflow(emulator: Emulator):
    emulator.registers[0].set(-(2**31))
    emulator.registers[1].set(1)
    assert emulator.SUB(0, 1, 2) == int(
        "100001" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 2**31 - 1


def test_mul_no_overflow(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(10)
    assert emulator.MUL(0, 1, 2) == int(
        "100010" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 50


def test_mul_overflow(emulator: Emulator):
    emulator.registers[0].set(2**30 - 1)
    emulator.registers[1].set(2)
    assert emulator.MUL(0, 1, 2) == int(
        "100010" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 2**31 - 2


def test_div(emulator: Emulator):
    emulator.registers[0].set(10)
    emulator.registers[1].set(5)
    assert emulator.DIV(0, 1, 2) == int(
        "100011" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 2


def test_and(emulator: Emulator):
    emulator.registers[0].set(int("001010", base=2))
    emulator.registers[1].set(int("000110", base=2))
    assert emulator.AND(0, 1, 2) == int(
        "101000" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == int("000010", base=2)


def test_or(emulator: Emulator):
    emulator.registers[0].set(int("001010", base=2))
    emulator.registers[1].set(int("000110", base=2))
    assert emulator.OR(0, 1, 2) == int(
        "101001" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == int("001110", base=2)


def test_xor(emulator: Emulator):
    emulator.registers[0].set(int("001010", base=2))
    emulator.registers[1].set(int("000110", base=2))
    assert emulator.XOR(0, 1, 2) == int(
        "101010" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == int("001100", base=2)


def test_cmpeq_true(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(5)
    assert emulator.CMPEQ(0, 1, 2) == int(
        "100100" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 1


def test_cmpeq_false(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(10)
    assert emulator.CMPEQ(0, 1, 2) == int(
        "100100" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 0


def test_cmplt_true(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(10)
    assert emulator.CMPLT(0, 1, 2) == int(
        "100101" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 1


def test_cmplt_false(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(5)
    assert emulator.CMPLT(0, 1, 2) == int(
        "100101" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 0


def test_cmple_true(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(5)
    assert emulator.CMPLE(0, 1, 2) == int(
        "100110" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 1


def test_cmple_false(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(4)
    assert emulator.CMPLE(0, 1, 2) == int(
        "100110" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 0


def test_shl_no_overflow(emulator: Emulator):
    emulator.registers[0].set(5)
    emulator.registers[1].set(1)
    assert emulator.SHL(0, 1, 2) == int(
        "101100" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 10


def test_shl_overflow(emulator: Emulator):
    emulator.registers[0].set(2**30)
    emulator.registers[1].set(1)
    assert emulator.SHL(0, 1, 2) == int(
        "101100" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == -(2**31)


def test_shr(emulator: Emulator):
    emulator.registers[0].set(10)
    emulator.registers[1].set(1)
    assert emulator.SHR(0, 1, 2) == int(
        "101101" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 5


def test_sra_positive(emulator: Emulator):
    emulator.registers[0].set(2**30)
    emulator.registers[1].set(1)
    assert emulator.SRA(0, 1, 2) == int(
        "101110" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == 2**29


def test_sra_negative(emulator: Emulator):
    emulator.registers[0].set(-(2**31))
    emulator.registers[1].set(1)
    assert emulator.SRA(0, 1, 2) == int(
        "101110" + "00010" + "00000" + "00001" + "00000000000", base=2
    )
    assert emulator.registers[2].get() == -(2**31) + 2**30


def test_addc_no_overflow(emulator: Emulator):
    emulator.registers[0].set(5)
    literal = 10
    assert emulator.ADDC(0, literal, 2) == int(
        "110000" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 15


def test_addc_overflow(emulator: Emulator):
    emulator.registers[0].set(2**31 - 1)
    literal = 1
    assert emulator.ADDC(0, literal, 2) == int(
        "110000" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == -(2**31)


def test_subc_no_overflow(emulator: Emulator):
    emulator.registers[0].set(10)
    literal = 5
    assert emulator.SUBC(0, literal, 2) == int(
        "110001" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 5


def test_subc_overflow(emulator: Emulator):
    emulator.registers[0].set(-(2**31))
    literal = 1
    assert emulator.SUBC(0, literal, 2) == int(
        "110001" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 2**31 - 1


def test_mulc_no_overflow(emulator: Emulator):
    emulator.registers[0].set(5)
    literal = 2
    assert emulator.MULC(0, literal, 2) == int(
        "110010" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 10


def test_mulc_overflow(emulator: Emulator):
    emulator.registers[0].set(2**30)
    literal = 2
    assert emulator.MULC(0, literal, 2) == int(
        "110010" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == -(2**31)


def test_divc(emulator: Emulator):
    emulator.registers[0].set(10)
    literal = 2
    assert emulator.DIVC(0, literal, 2) == int(
        "110011" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 5


def test_andc(emulator: Emulator):
    emulator.registers[0].set(int("101010101010", base=2))
    literal = int("101000000110", base=2)
    assert emulator.ANDC(0, literal, 2) == int(
        "111000" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == int("101000000010", base=2)


def test_orc(emulator: Emulator):
    emulator.registers[0].set(int("101010101010", base=2))
    literal = int("101000000110", base=2)
    assert emulator.ORC(0, literal, 2) == int(
        "111001" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == int("101010101110", base=2)


def test_xorc(emulator: Emulator):
    emulator.registers[0].set(int("101010101010", base=2))
    literal = int("101000000110", base=2)
    assert emulator.XORC(0, literal, 2) == int(
        "111010" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == int("000010101100", base=2)


def test_cmpeqc_true(emulator: Emulator):
    emulator.registers[0].set(5)
    literal = 5
    assert emulator.CMPEQC(0, literal, 2) == int(
        "110100" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 1


def test_cmpeqc_false(emulator: Emulator):
    emulator.registers[0].set(5)
    literal = 10
    assert emulator.CMPEQC(0, literal, 2) == int(
        "110100" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 0


def test_cmpltc_true(emulator: Emulator):
    emulator.registers[0].set(5)
    literal = 10
    assert emulator.CMPLTC(0, literal, 2) == int(
        "110101" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 1


def test_cmpltc_false(emulator: Emulator):
    emulator.registers[0].set(5)
    literal = 5
    assert emulator.CMPLTC(0, literal, 2) == int(
        "110101" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 0


def test_cmplec_true(emulator: Emulator):
    emulator.registers[0].set(5)
    literal = 5
    assert emulator.CMPLEC(0, literal, 2) == int(
        "110110" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 1


def test_cmplec_false(emulator: Emulator):
    emulator.registers[0].set(5)
    literal = 4
    assert emulator.CMPLEC(0, literal, 2) == int(
        "110110" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 0


def test_shlc_no_overflow(emulator: Emulator):
    emulator.registers[0].set(5)
    literal = 1
    assert emulator.SHLC(0, literal, 2) == int(
        "111100" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 10


def test_shlc_overflow(emulator: Emulator):
    emulator.registers[0].set(2**30)
    literal = 1
    assert emulator.SHLC(0, literal, 2) == int(
        "111100" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == -(2**31)


def test_shrc(emulator: Emulator):
    emulator.registers[0].set(10)
    literal = 1
    assert emulator.SHRC(0, literal, 2) == int(
        "111101" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 5


def test_srac_positive(emulator: Emulator):
    emulator.registers[0].set(2**30)
    literal = 1
    assert emulator.SRAC(0, literal, 2) == int(
        "111110" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 2**29


def test_srac_negative(emulator: Emulator):
    emulator.registers[0].set(-(2**31))
    literal = 1
    assert emulator.SRAC(0, literal, 2) == int(
        "111110" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == -(2**31) + 2**30


def test_ld(emulator: Emulator):
    emulator.data_memory.set(0x5C, 1337)
    emulator.registers[0].set(0x50)
    literal = 0xC
    assert emulator.LD(0, literal, 2) == int(
        "011000" + "00010" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[2].get() == 1337


def test_st(emulator: Emulator):
    emulator.registers[0].set(0x40)
    literal = 0xD
    emulator.registers[1].set(1337)
    assert emulator.ST(1, literal, 0) == int(
        "011001" + "00001" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.data_memory.get(emulator.registers[0].get() + literal) == 1337


def test_jmp(emulator: Emulator):
    emulator.registers[0].set(24)
    assert emulator.JMP(0, 1) == int(
        "011011" + "00001" + "00000" + build_bin_string(0), base=2
    )
    assert emulator.registers[1].get() == 4
    assert emulator.program_counter == 24


def test_jmp_mask_msb(emulator: Emulator):
    emulator.registers[0].set(-(2**31))
    assert emulator.JMP(0, 1) == int(
        "011011" + "00001" + "00000" + build_bin_string(0), base=2
    )
    assert emulator.registers[1].get() == 4
    assert emulator.program_counter == 0


def test_beq_true(emulator: Emulator):
    emulator.registers[0].set(0)
    label = 24
    literal = (label - (emulator.program_counter + 4)) // emulator.INSTR_WIDTH
    assert emulator.BEQ(0, label, 1) == int(
        "011101" + "00001" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[1].get() == 4
    assert emulator.program_counter == 24


def test_beq_false(emulator: Emulator):
    emulator.registers[0].set(1)
    label = 24
    literal = (label - (emulator.program_counter + 4)) // emulator.INSTR_WIDTH
    assert emulator.BEQ(0, label, 1) == int(
        "011101" + "00001" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[1].get() == 4
    assert emulator.program_counter == 4


def test_bne_true(emulator: Emulator):
    emulator.registers[0].set(1)
    label = 24
    literal = (label - (emulator.program_counter + 4)) // emulator.INSTR_WIDTH
    assert emulator.BNE(0, label, 1) == int(
        "011110" + "00001" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[1].get() == 4
    assert emulator.program_counter == 24


def test_bne_false(emulator: Emulator):
    emulator.registers[0].set(0)
    label = 24
    literal = (label - (emulator.program_counter + 4)) // emulator.INSTR_WIDTH
    assert emulator.BNE(0, label, 1) == int(
        "011110" + "00001" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[1].get() == 4
    assert emulator.program_counter == 4


def test_ldr(emulator: Emulator):
    label = 24
    literal = (label - (emulator.program_counter + 4)) // emulator.INSTR_WIDTH
    emulator.data_memory.set(label, 1337)
    assert emulator.LDR(label, 0) == int(
        "011111" + "00000" + "00000" + build_bin_string(literal), base=2
    )
    assert emulator.registers[0].get() == 1337


def test_halt(emulator: Emulator):
    assert emulator.HALT() == 0
    assert emulator.program_counter == 0

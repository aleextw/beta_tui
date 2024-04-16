from app import EmulatorApp
from emulator import Emulator
import argparse
from rich import print
import pathlib
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Beta Emulator V2",
        description="Assembler, Disassembler, and Emulator for the Beta ISA taught during 50.002 Computation Structures at SUTD.",
    )

    parser.add_argument("-i", "--instr")
    parser.add_argument("-d", "--data")
    parser.add_argument("-x", "--hex", action="store_true")
    parser.add_argument("-e", "--emulate", action="store_true")

    args = parser.parse_args()

    if args.emulate:
        app = EmulatorApp()
        app.run()
    elif args.instr is None:
        print(
            "[bold red]Error:[/bold red] Please provide the path to the instruction data with the [italic blue]-i[/italic blue] flag, and the path to the memory data with the [italic blue]-d[/italic blue] flag."
        )
        print("[bold yellow]Relative filepaths can be used.[/bold yellow]")
    else:
        hex = False

        if not args.hex:
            print(
                "[bold yellow]Info:[/bold yellow] Hex flag not provided, defaulting to binary output."
            )
        else:
            hex = True

        emu = Emulator()
        instr_mem_filepath = pathlib.Path(args.instr)
        if args.data is not None:
            data_mem_filepath = pathlib.Path(args.data)
        output_dir = instr_mem_filepath.parent

        print(
            f"[bold yellow]Info:[/bold yellow] Loading instruction memory at {instr_mem_filepath}"
        )
        print(
            f"[bold yellow]Info:[/bold yellow] Loading data memory at {data_mem_filepath}"
        )
        print(f"[bold yellow]Info:[/bold yellow] Writing data to {output_dir}")

        if not instr_mem_filepath.is_file():
            print("[bold red]Error:[/bold red] The specified file does not exist.")
        else:
            if not data_mem_filepath.is_file():
                print(
                    f"[bold yellow]Info:[/bold yellow] Failed to find data memory at {data_mem_filepath}, parsing only instruction memory"
                )
                emu.load_files(instr_mem_filepath)
            else:
                emu.load_files(instr_mem_filepath, data_mem_filepath)

            instr_mem = emu.instruction_memory
            serialized_instr_mem = [
                ("h" if hex else "b")
                + str(instruction.hex if hex else instruction.bin)
                + "\n"
                for instruction in instr_mem.data
            ]

            data_mem = emu.data_memory
            serialized_data_mem = [
                ("h" if hex else "b") + str(data.hex if hex else data.bin) + "\n"
                for data in data_mem.data
            ]

            output_instr_mem_path = instr_mem_filepath.with_suffix(
                ".hex" if hex else ".bin"
            )
            print(
                f"[bold yellow]Info:[/bold yellow] Writing instruction to {output_instr_mem_path}"
            )
            with open(output_instr_mem_path, "w+") as file:
                file.writelines(tqdm(serialized_instr_mem))

            if args.data:
                output_data_mem_path = data_mem_filepath.with_suffix(
                    ".hex" if hex else ".bin"
                )
                print(
                    f"[bold yellow]Info:[/bold yellow] Writing memory to {output_data_mem_path}"
                )
                with open(output_data_mem_path, "w+") as file:
                    file.writelines(tqdm(serialized_data_mem))

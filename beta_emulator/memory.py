import logging
import math
from bitstring import BitArray
from tqdm import tqdm


class Memory:
    """
    Represents a memory unit to be emulated
    """

    def __init__(self, size: int, width: int, name: str):
        """
        size: int - Size of the memory segment in bytes
        width: int - Width of each memory word in bytes
        name: str - Name to give the Memory instance for debugging purposes
        """
        self.size = size
        self.width = width
        self.name = name
        self.write_mask = int("1" * self.width * 8, base=2)
        self.data = [
            BitArray(int=0, length=width * 8) for _ in tqdm(range(size // width))
        ]

        self.unparsed_data = [None for _ in tqdm(range(size // width))]

    def get(self, address: int) -> int:
        """
        Returns the data starting at byte position 'address'
        The lower log(width) bits of the address will be masked
        """
        self._check_address_validity(address)

        return self.data[address >> math.ceil(math.log2(self.width))].int

    def set(self, address: int, data: int):
        """
        Set the memory word starting at byte position 'address'
        The lower log(width) bits of the address will be masked
        """
        self._check_address_validity(address)

        masked_val = data & self.write_mask
        if masked_val < 0:
            data = BitArray(int=masked_val, length=self.width * 8)
        else:
            data = BitArray(uint=masked_val, length=self.width * 8)

        self.data[address >> math.ceil(math.log2(self.width))] = data

    def get_instr(self, address: int) -> list:
        """
        Returns the instruction starting at byte position 'address'
        The lower log(width) bits of the address will be masked
        """
        self._check_address_validity(address)

        return self.unparsed_data[address >> math.ceil(math.log2(self.width))]

    def set_instr(self, address: int, data: list):
        """
        Sets the instruction starting at byte position 'address'
        The lower log(width) bits of the address will be masked
        """
        self._check_address_validity(address)

        self.unparsed_data[address >> math.ceil(math.log2(self.width))] = data

    def _check_address_validity(self, address: int):
        """
        Check whether address is trying to access < 0 or > size
        """
        if address + self.width > self.size or address < 0:
            error_message = (
                f"Tried to access memory {self.name} at out of "
                f"bounds address {address} (Memory size {self.size})"
            )
            logging.error(error_message)
            raise IndexError(error_message)

from bitstring import BitArray


class Register:
    """
    Represents a register to be emulated

    Methods:
    get() - Returns the stored contents of the register
    set(data) - Sets the register contents to the specified data
              - If read_only, ignores data
    """

    def __init__(self, width: int, read_only: bool, name: str):
        """
        width: int - Width of the register contents in bytes
        name: str - Name to give the Register instance for debugging purposes
        """
        self.width = width
        self.write_mask = int("1" * self.width * 8, base=2)
        self.read_only = read_only
        self.name = name

        self.data = BitArray(int=0, length=self.width * 8)

    def get(self):
        """
        Returns the stored contents of the register
        """
        return self.data.int

    def set(self, data: int):
        """
        Sets the register contnts to the specified data
        If read_only, ignores data
        """
        if not self.read_only:
            masked_val = data & self.write_mask
            if masked_val < 0:
                self.data.int = masked_val
            else:
                self.data.uint = masked_val

    def reset(self):
        """
        Reset contents of register
        """
        self.data.int = 0

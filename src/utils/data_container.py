class Table:
    def __init__(self):
        # Expect key in the form of (key_1, key_2) or (key_2, key_1)
        self.__table = {}

    def __setitem__(self, key: tuple, value: any) -> None:
        assert isinstance(key, tuple) and len(
            key) == 2, f"Invalid key (expect (key_1, key_2)): {key}"

        self.__table[tuple(sorted(key))] = value

    def __getitem__(self, key: tuple) -> any:
        """
        Return None if the key does not exist in the table

        :param key: Key of the target cell
        :type key: tuple
        :return: The value of the cell
        :rtype: Any
        """
        assert isinstance(key, tuple) and len(
            key) == 2, f"Invalid key (expect (key_1, key_2)): {key}"

        if tuple(sorted(key)) in self.__table:
            return self.__table[tuple(sorted(key))]
        else:
            return None

    # TODO: Maybe add a delete method if have time (not necessary at the current stage)

    def has_key(self, key: tuple) -> bool:
        if not isinstance(key, tuple) or not len(key) == 2:
            # Not a possible key
            return False

        return tuple(sorted(key)) in self.__table

    def clear(self) -> None:
        """
        Reset to default table, i.e. an empty table
        """
        self.__table = {}

    @property
    def table(self) -> dict:
        return self.__table

    @property
    def keys(self):
        # TODO: See if this can be changed to a better approach, e.g. inherit the class Iterable in Python instead of defining my own property
        return self.__table.keys()

    @property
    def values(self):
        # TODO: See if this can be changed to a better approach, e.g. inherit the class Iterable in Python instead of defining my own property
        return self.__table.values()

    @property
    def items(self):
        # TODO: See if this can be changed to a better approach, e.g. inherit the class Iterable in Python instead of defining my own property
        return self.__table.items()

    def __repr__(self) -> str:
        # TODO: Add test to test the correctness of below output format

        is_first_cell = True

        table_text = f"{{"
        for key, value in self.__table.items():
            if is_first_cell:
                is_first_cell = False
            else:
                table_text += f", \n"

            table_text += f"\"{tuple(sorted(key))}\": {value}"

        table_text += f"}}"

        return table_text

    @classmethod
    def combine(cls, *tables):
        combined_table = Table()

        for table in tables:
            # TODO: Confirm the case (will overwriting happen? )
            combined_table.table.update(table.table)

        return combined_table

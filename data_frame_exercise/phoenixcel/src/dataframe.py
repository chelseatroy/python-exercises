import csv
from phoenixcel.src.series import Series
from phoenixcel.src.groupby import GroupBy

class DataFrame():
    def __init__(self):
        self._dictionary = {}
        self._list = []

    # Ways to crate an instance
    @classmethod
    def from_csv(cls, file_path):
        df = cls()
        header_unread = True

        with open(file_path) as f:
            reader = csv.DictReader(f)

            for row in reader:
                if header_unread:
                    for key in row.keys():
                        df._dictionary[key] = Series()
                    header_unread = False
                
                df._list.append(row)

                for key in row.keys():
                    df._dictionary[key].append(row[key])

            for key in list(df._dictionary.keys()):
                setattr(df, key.lower().replace(" ", "_"), df._dictionary[key])
        return df

    @classmethod
    def from_rows(cls, rows):
        df = cls()

        if not rows:
            return df
        
        for key in rows[0].keys():
            df._dictionary[key] = Series()
        for row in rows:
            for key in rows[0].keys():
                df._dictionary[key].append(row[key])
        
            df._list.append(row)

        for key in list(df._dictionary.keys()):
            setattr(df, key.lower().replace(" ", "_"), df._dictionary[key])

        return df

    @classmethod
    def from_dictionary(cls, dictionary):
        df = cls()

        if not dictionary:
            return df
        
        df._dictionary = dictionary
        for i in range(len(dictionary[list(dictionary.keys())[0]])):
            item = {}
            for key in dictionary.keys():
                item[key] = dictionary[key][i]
            df._list.append(item)

        for key in list(df._dictionary.keys()):
            setattr(df, key.lower().replace(" ", "_"), df._dictionary[key])

        return df

    # Properties
    @property
    def shape(self):
        return \
            len(self._dictionary.keys()), \
            len(self._dictionary[list(self._dictionary.keys())[0]])

    @property
    def columns(self):
        return list(self._dictionary.keys())

    # Methods for getting a column in the dictionary
    def __getitem__(self, item):
        '''
        Get a reference to a column in the dataframe.

        Input:
          item - the column header

        Output:
          the column, which is a series

        Modifies:
          Nothing
        '''
        return Series(self._dictionary[item])

    # Method for setting a column in the dictionary
    def __setitem__(self, key, value):
        '''
        Set a new column in the dataframe.

        Inputs:
          key - the column header
          value - the column (as a Series for consistency, please)

        Outputs:
          None

        Modifies:
          Modifies the dataframe object in place.
        '''
        self._dictionary[key] = value
        for index, item in enumerate(self._list):
            item[key] = value[index]

        setattr(self, key.lower().replace(" ", "_"), self._dictionary[key])

    def where(self, condition):
        rows = [row for row in self._list if condition(row)]
        return DataFrame.from_rows(rows)

    def assign(self, **kwargs):
        for key, value in kwargs.items():
            new_column = Series()
            for row in self._list:
                new_column.append(value(row))
            self.__setitem__(key, new_column)
        return self

    def group_by(self, column):
        '''
        Returns an object that aggregates the items in the dataframe
        based on one value that they have in common,
        similar to a pivot table in the software to which
        phoenixcell's name pays tribute (Please don't sue me, Microsoft)

        Inputs:
          column - the column on whose value the items should be grouped

        Outputs:
          A new GroupBy() object

        Modifies:
          Nothing
        '''
        groups = GroupBy()
        for item in self._list:
            maybe_unique_column_value = item[column]
            if maybe_unique_column_value in groups.keys():
                groups[maybe_unique_column_value].append(item)
            else:
                groups[maybe_unique_column_value] = Series()
                groups[maybe_unique_column_value].append(item)
        return groups
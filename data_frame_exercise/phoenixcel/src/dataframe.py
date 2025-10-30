import csv
from phoenixcel.src.series import Series
from phoenixcel.src.groupby import GroupBy

class DataFrame():
    def __init__(self):
        self._list = []

    # Ways to crate an instance
    @classmethod
    def from_csv(cls, file_path):
        df = cls()
        header_unread = True

        with open(file_path) as f:
            reader = csv.DictReader(f)

            for row in reader:                
                df._list.append(row)

        return df

    @classmethod
    def from_rows(cls, rows):
        df = cls()

        if not rows:
            return df
        
        for row in rows:
            df._list.append(row)

        return df

    @classmethod
    def from_dictionary(cls, dictionary):
        df = cls()

        if not dictionary:
            return df
        
        for i in range(len(dictionary[list(dictionary.keys())[0]])):
            item = {}
            for key in dictionary.keys():
                item[key] = dictionary[key][i]
            df._list.append(item)

        return df

    # Properties
    @property
    def shape(self):
        if not self._list:
            return (0, 0)
        
        #assumes all rows have same number of columns as first row 
        return len(self._list[0]), len(self._list)

    @property
    def columns(self):
        if not self._list:
            return []
        
        return list(self._list[0].keys()) #assumes all rows have values in the same columns as the first row

    # Methods for getting a column in the dictionary
    def __getitem__(self, item):
        '''
        Get a reference to a column in the dataframe.

        Input:
          item - the column header

        Output:
          the column as a Series

        Modifies:
          Nothing
        '''
        return Series([row[item] for row in self._list])

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
        if not len(value) == len(self._list):
            raise ValueError("Length of new column must match number of rows in dataframe")

        for index, row in enumerate(self._list):
            row[key] = value[index]

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
from copy import deepcopy

class Table():
    def __init__(self):
        self.data = {}
        self._transaction_data = {}
        self.transaction_in_progress = False

    def add_column(self, name):
        if not self.transaction_in_progress:
            raise Exception("You must be inside a transaction to write to this table.")

        if name in self._transaction_data.keys():
            raise Exception("Column already exists.")

        self._transaction_data[name] = []

    def drop_column(self, name):
        if not self.transaction_in_progress:
            raise Exception("You must be inside a transaction to write to this table.")

        # Cannot delete a nonexistent column
        if name not in self._transaction_data.keys():
            raise Exception("Column does not exist.")

        self._transaction_data.pop(name)

    def update_column(self, name, value):
        if not self.transaction_in_progress:
            raise Exception("You must be inside a transaction to write to this table.")

        self._transaction_data[name] = value

    def begin_transaction(self):
        """
        Starts a transaction in the table.
        """
        self._transaction_data = deepcopy(self.data)

        # Indicate to the table that there is a transaction in progress.
        self.transaction_in_progress = True

    def commit_transaction(self):
        """
        Completes a transaction in the database and saves the changes.
        """
        # If there is no transaction in progress, there is nothing to complete.
        if not self.transaction_in_progress:
            raise Exception("No transaction in progress.")

        # Save the copied and modified transaction_data back to the "main" data variable
        self.data = self._transaction_data

        # Empty out the transaction data and
        self._transaction_data = {}
        # Flip the indicator that there's a transaction in progress.
        self.transaction_in_progress = False

    def roll_back_transaction(self):
        """
        Cancels a transaction in the database and discards the changes.
        """
        # If there is no transaction in progress, there is nothing to cancel.
        if not self.transaction_in_progress:
            raise Exception("No transaction in progress.")

        # Empty out the transaction_data and
        self._transaction_data = {}
        # Flip the indicator that there's a transaction in progress.
        self.transaction_in_progress = False
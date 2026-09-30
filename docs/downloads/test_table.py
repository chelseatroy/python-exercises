import unittest
from table import Table


class TestTableInitialization(unittest.TestCase):
    def test_initialization(self):
        table = Table()
        self.assertEqual(table.data, {})
        self.assertFalse(table.transaction_in_progress)


class TestTableTransactions(unittest.TestCase):
    def setUp(self):
        self.table = Table()

    def test_commit_transaction_without_begin_raises_exception(self):
        with self.assertRaises(Exception) as context:
            self.table.commit_transaction()
        self.assertEqual(str(context.exception), "No transaction in progress.")

    def test_commit_transaction_saves_changes(self):
        self.table.begin_transaction()
        self.table.update_column("new_col", [1, 2, 3])
        self.table.commit_transaction()

        self.assertEqual(self.table.data, {"new_col": [1, 2, 3]})
        self.assertFalse(self.table.transaction_in_progress)

    def test_roll_back_transaction_without_begin_raises_exception(self):
        with self.assertRaises(Exception) as context:
            self.table.roll_back_transaction()
        self.assertEqual(str(context.exception), "No transaction in progress.")

    def test_roll_back_transaction_discards_changes(self):
        self.table.data = {"col1": [1, 2, 3]}
        self.table.begin_transaction()
        self.table.update_column("col1", [4, 5, 6])
        self.table.roll_back_transaction()

        self.assertEqual(self.table.data, {"col1": [1, 2, 3]})
        self.assertFalse(self.table.transaction_in_progress)


class TestTableAddColumn(unittest.TestCase):
    def setUp(self):
        self.table = Table()

    def test_add_column_without_transaction_raises_exception(self):
        with self.assertRaises(Exception) as context:
            self.table.add_column("col1")
        self.assertEqual(str(context.exception), "You must be inside a transaction to write to this table.")

    def test_add_column_successfully(self):
        self.table.begin_transaction()
        self.table.add_column("col1")
        self.table.commit_transaction()

        self.assertEqual(self.table.data["col1"], [])

    def test_add_column_that_already_exists_raises_exception(self):
        self.table.begin_transaction()
        self.table.add_column("col1")

        with self.assertRaises(Exception) as context:
            self.table.add_column("col1")
        self.assertEqual(str(context.exception), "Column already exists.")

    def test_add_column_and_commit(self):
        self.table.begin_transaction()
        self.table.add_column("col1")
        self.table.commit_transaction()

        self.assertEqual(self.table.data, {"col1": []})


class TestTableDropColumn(unittest.TestCase):
    def setUp(self):
        self.table = Table()

    def test_drop_column_without_transaction_raises_exception(self):
        with self.assertRaises(Exception) as context:
            self.table.drop_column("col1")
        self.assertEqual(str(context.exception), "You must be inside a transaction to write to this table.")

    def test_drop_column_that_doesnt_exist_raises_exception(self):
        self.table.begin_transaction()

        with self.assertRaises(Exception) as context:
            self.table.drop_column("col2")
        self.assertEqual(str(context.exception), "Column does not exist.")

    def test_drop_column_and_rollback_doesnt_drop_column(self):
        self.table.data = {"col1": [1, 2, 3]}
        self.table.begin_transaction()
        self.table.drop_column("col1")
        self.table.roll_back_transaction()

        self.assertEqual(self.table.data.get("col1"), [1, 2, 3])

    def test_drop_column_and_commit(self):
        self.table.data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        self.table.begin_transaction()
        self.table.drop_column("col1")
        self.table.commit_transaction()

        self.assertNotIn("col1", self.table.data)

class TestTableUpdateColumn(unittest.TestCase):
    def setUp(self):
        self.table = Table()

    def test_update_column_without_transaction_raises_exception(self):
        with self.assertRaises(Exception) as context:
            self.table.update_column("col1", [1, 2, 3])
        self.assertEqual(str(context.exception), "You must be inside a transaction to write to this table.")

    def test_update_column_successfully(self):
        self.table.begin_transaction()
        self.table.update_column("col1", [1, 2, 3])
        self.table.commit_transaction()

        self.assertEqual(self.table.data["col1"], [1, 2, 3])

    def test_update_column_and_commit(self):
        self.table.data = {"col1": [1, 2, 3]}
        self.table.begin_transaction()
        self.table.update_column("col1", [4, 5, 6])
        self.table.commit_transaction()

        self.assertEqual(self.table.data, {"col1": [4, 5, 6]})


class TestTableIntegration(unittest.TestCase):
    def setUp(self):
        self.table = Table()

    def test_multiple_operations_in_transaction(self):
        self.table.begin_transaction()
        self.table.add_column("col1")
        self.table.update_column("col1", [1, 2, 3])
        self.table.add_column("col2")
        self.table.update_column("col2", [4, 5, 6])
        self.table.commit_transaction()

        self.assertEqual(self.table.data, {"col1": [1, 2, 3], "col2": [4, 5, 6]})

    def test_multiple_transactions(self):
        # First transaction
        self.table.begin_transaction()
        self.table.add_column("col1")
        self.table.commit_transaction()

        # Second transaction
        self.table.begin_transaction()
        self.table.update_column("col1", [1, 2, 3])
        self.table.commit_transaction()

        self.assertEqual(self.table.data, {"col1": [1, 2, 3]})

    def test_rollback_does_not_affect_committed_data(self):
        self.table.begin_transaction()
        self.table.add_column("col1")
        self.table.commit_transaction()

        self.table.begin_transaction()
        self.table.add_column("col2")
        self.table.roll_back_transaction()

        self.assertEqual(self.table.data, {"col1": []})
        self.assertNotIn("col2", self.table.data)

import sys
class TestTableMemPerformance(unittest.TestCase):
    def setUp(self):
        self.table = Table()
        self.table.begin_transaction()
        self.table.update_column("multiples_of_1", [1, 2, 3, 4, 5])
        self.table.update_column("multiples_of_2", [2, 4, 6, 8, 10])
        self.table.update_column("multiples_of_3", [3, 6, 9, 12, 15])
        self.table.commit_transaction()

    def calculate_table_space(self):
        def get_bytes(dictionary):
            return sys.getsizeof(str(dictionary))
        
        return get_bytes(self.table.data) + get_bytes(self.table._transaction_data)

    def test_transaction_remains_beneath_acceptable_memory_growth_factor(self):
        initial_space = self.calculate_table_space()

        self.table.begin_transaction()

        mid_space = self.calculate_table_space()

        self.table.commit_transaction()

        final_space = self.calculate_table_space()

        # Assert space used during transaction does not exceed 10%
        # on top of initial space required for this table
        self.assertLess(mid_space, initial_space * 1.1)
        # Assert final space is not larger than initial
        self.assertLessEqual(final_space, initial_space)

if __name__ == "__main__":
    unittest.main()

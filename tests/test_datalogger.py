__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import unittest

import numpy as np

from deeptracking.utils.data_logger import DataLogger


class TestDatasetMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.df_label = "test"
        self.logger = DataLogger()
        self.logger.create_dataframe(self.df_label, ["A", "B"])

    def tearDown(self):
        pass

    def test_it_should_return_empty_list_of_df_at_init(self):
        data_logger = DataLogger()
        self.assertListEqual(data_logger.get_dataframes_id(), [])

    def test_it_should_add_dataframe(self):
        data_logger = DataLogger()
        data_logger.create_dataframe(self.df_label, ["A", "B"])
        self.assertTrue(self.df_label in data_logger.get_dataframes_id())

    def test_it_should_raise_error_if_input_data_size_is_wrong(self):
        self.assertRaises(IndexError, self.logger.add_row, self.df_label, [1])

    def test_it_should_add_data(self):
        truth = np.zeros((5, 2))
        for i in range(5):
            self.logger.add_row(self.df_label, [i, i+1])
            truth[i] = [i, i+1]
        np.testing.assert_almost_equal(self.logger.get_as_numpy(self.df_label), truth)

    def test_it_should_add_row_from_dict(self):
        data = {"B": 0, "A": 1}
        truth = np.zeros((1, 2))
        truth[0, 0] = 1
        self.logger.add_row_from_dict(self.df_label, data)
        np.testing.assert_almost_equal(self.logger.get_as_numpy(self.df_label), truth)

    def test_it_should_throw_exception_if_add_row_from_dict_missing_data(self):
        data = {"B": 0, "C": 1}
        truth = np.zeros((1, 2))
        truth[0, 0] = 1
        self.assertRaises(KeyError, self.logger.add_row_from_dict, self.df_label, data)

    def test_it_should_return_columns_in_order(self):
        cols = ["a", "A", "8", "test", "Z", "C"]
        name = "manycol"
        self.logger.create_dataframe(name, cols)
        col_names = self.logger.get_dataframe_columns(name)
        self.assertListEqual(col_names, cols)

    def test_simple_load_save_test(self):
        """
        Should use mock object but this is simple and efficient...
        :return:
        """
        test_path = "data"
        self.logger.create_dataframe("other", ["col1"])
        self.logger.create_dataframe("other2", ["col1", "col2", "col3", "col4"])
        for i in range(5):
            self.logger.add_row("other", [0.01 + i])
            self.logger.add_row(self.df_label, [i, i+1])
        self.logger.save(test_path)

        load_logger = DataLogger()
        load_logger.load(test_path)
        np.testing.assert_almost_equal(self.logger.get_as_numpy(self.df_label), load_logger.get_as_numpy(self.df_label))
        np.testing.assert_almost_equal(self.logger.get_as_numpy("other"), load_logger.get_as_numpy("other"))
        np.testing.assert_almost_equal(self.logger.get_as_numpy("other2"), load_logger.get_as_numpy("other2"))
        load_logger.clear_csv(test_path)

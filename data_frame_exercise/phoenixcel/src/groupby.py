class GroupBy(dict):
    def sum(self, on=None):
        return self.aggregate(on=on, using_func=sum)

    def average(self, on=None):
        def func(listo):
            return sum(listo) / len(listo)

        return self.aggregate(on=on, using_func=func)

    avg = average

    def count(self, on=None):
        return self.aggregate(on=on, using_func=len)

    def min(self, on=None):
        return self.aggregate(on=on, using_func=min)

    def max(self, on=None):
        return self.aggregate(on=on, using_func=max)

    def spread(self, on=None):
        def func(listo):
            return max(listo) - min(listo)

        return self.aggregate(on=on, using_func=func)

    def aggregate(self, on=None, using_func=None):
        aggregator = {}
        if on == None:
            raise Exception("What column do you want aggregated?")
        if using_func == None:
            raise Exception(f"How do you want '{on}' aggregated?")
        else:
            for key in self.keys():
                addends = [item[on] for item in self[key]]
                aggregator[key] = using_func(addends)
        return aggregator

    def describe_with(self, *args):
        descriptions = {}
        for aggregation in args:
            if aggregation['agg'] == 'aggregate':
                result = self.aggregate(on=aggregation['column'], using_func=aggregation['using_func'])
                function_name = aggregation['using_func'].__name__
            else:
                aggregation_function = getattr(self, aggregation['agg'])
                result = aggregation_function(on=aggregation['column'])
                function_name = aggregation['agg']

            for result_key in result.keys():
                if not descriptions.get(result_key):
                    descriptions[result_key] = {}
                aggregation_label = f"{aggregation['column']}_{function_name}"
                descriptions[result_key][aggregation_label] = result[result_key]
        return GroupBy(descriptions)

    def print_cute(self):
        '''
        Prints out a GroupBy or a GroupBy Description in a nice format.

        Input:
          None - this method operates on the existing GroupBy object.

        Output:
          None - no return value

        Modifies:
          Prints to standard out with a list of the groups. For each group,
          prints an indented list of the items in it (for a GroupBy),
          or an indented list of summary statistics (for a Groupby Description).
        '''
        for key, value in self.items():
            print(key)

            if isinstance(value, dict):
                for key, component in value.items():
                    print(f"   {key} : {component}")
            else:
                for component in value:
                    print(f"  {component}")



class GenericTest:

    hr = '―' * 78

    def __init__(self):
        print(self.hr)

    def setup_class(self):
        pass

    def teardown_class(self):
        pass

    def run_all(self):
        self.setup_class()
        for item in self.__dir__():
            if item.startswith('test'):
                print(f"Running test: {item}")
                self.__getattribute__(item)()
        self.teardown_class()

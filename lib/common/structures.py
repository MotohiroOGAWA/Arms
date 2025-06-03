
class safedict(dict):
    def __missing__(self, key):
        return None

class NamedField(str):
    """
    A string-like object that also carries a type annotation.
    """

    def __new__(cls, name: str, type_: str):
        obj = super().__new__(cls, name)
        obj.type = type_
        return obj

    def __repr__(self):
        return f"{super().__repr__()}:{self.type}"

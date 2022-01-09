

def is_named_tuple(x):
    """Returns whether an object is an instance of a collections.namedtuple.

    Examples::
      is_named_tuple((42, 'hi')) ==> False
      Foo = collections.namedtuple('Foo', ['a', 'b'])
      is_named_tuple(Foo(a=42, b='hi')) ==> True

    Args:
      x: The object to check.
    """
    return isinstance(x, tuple) and hasattr(x, "_fields") and hasattr(x, "_asdict")

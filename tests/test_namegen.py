from mrfoptools.namegen import generate_name


def test_generate_name():
    name = generate_name()
    assert isinstance(name, str)
    assert len(name) > 0
    assert '-' in name
    assert len(name.split('-')) == 2
    
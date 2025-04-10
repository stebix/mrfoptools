from mrfoptools.namegen import generate_name, NameElement


class Test_generate_name:

    def test_generate_with_two_elements(self):
        order = (
            NameElement.ADJECTIVE,
            NameElement.NOUN
        )
        name = generate_name(order)
        print(name)
        assert isinstance(name, str)
        assert len(name) > 0
        assert '-' in name
        assert len(name.split('-')) == 2


    def test_generate_with_three_elements(self):
        order = (
            NameElement.ADJECTIVE,
            NameElement.NOUN,
            NameElement.ANIMAL
        )
        name = generate_name(order)
        print(name)
        assert isinstance(name, str)
        assert len(name) > 0
        assert '-' in name
        assert len(name.split('-')) == 3

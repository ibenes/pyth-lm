import unittest
import vocab

class IndexGeneratorTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_zero_start(self):
        ig = vocab.IndexGenerator([])
        self.assertEqual(ig.next(), 0)

    def test_respects_assigned(self):
        ig = vocab.IndexGenerator([0, 1])
        self.assertEqual(ig.next(), 2)
        
    def test_respects_sparse_assigned(self):
        ig = vocab.IndexGenerator([0, 2, 3])
        nexts = [ig.next() for i in range(2)]
        self.assertEqual(nexts, [1, 4])


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.example_line = "hello world of vocabularies !"

    def test_getitem(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        self.assertEqual(vocabulary['<unk>'], 0)

    def test_getitem_for_unknown(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        self.assertEqual(vocabulary['a'], 0)

    def test_addition_un_unks(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        self.assertEqual(vocabulary['world'], 0)

        vocabulary.add_from_text(self.example_line)
        self.assertNotEqual(vocabulary['world'], 0)

    def test_addition_is_unique(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        vocabulary.add_from_text(self.example_line)
        self.assertNotEqual(vocabulary['world'], vocabulary['hello'])
        self.assertNotEqual(vocabulary['of'], vocabulary['hello'])
        self.assertNotEqual(vocabulary['of'], vocabulary['!'])
        self.assertNotEqual(vocabulary['world'], vocabulary['!'])

    def test_lenght_of_empty(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        self.assertEqual(len(vocabulary), 1)

    def test_lenght_after_addition(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        vocabulary.add_from_text(self.example_line)
        self.assertEqual(len(vocabulary), 6)

    def test_add_word(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        vocabulary.add_word('hi')
        self.assertEqual(len(vocabulary), 2)

    def test_add_already_known_word(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        vocabulary.add_word('<unk>')
        self.assertEqual(len(vocabulary), 1)

    def test_backward_translation_consistent(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        vocabulary.add_word('hi')
        self.assertEqual(vocabulary.i2w(vocabulary['hi']), 'hi')
        

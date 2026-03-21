import unittest
from cash.notebook.analysis import CodeAnalyzer


class TestEnhancedDependencyDetection(unittest.TestCase):
    """Test enhanced dependency detection for dataframe and object modifications."""
    
    def test_subscript_assignment(self):
        """Test df['column'] = value is detected as both read and write of df"""
        code = "df['column'] = value"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        # df is both read (to access it) and written (modified)
        self.assertIn('df', inputs, "df should be in inputs when modifying column")
        self.assertIn('df', outputs, "df should be in outputs when modifying column")
        self.assertIn('value', inputs, "value should be in inputs")
    
    def test_subscript_read(self):
        """Test df['column'] only reads df"""
        code = "x = df['column']"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        self.assertIn('df', inputs, "df should be in inputs when reading column")
        self.assertNotIn('df', outputs, "df should NOT be in outputs when only reading")
        self.assertIn('x', outputs, "x should be in outputs")
    
    def test_loc_assignment(self):
        """Test df.loc[idx] = value is detected as modification"""
        code = "df.loc[idx, 'col'] = value"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        self.assertIn('df', inputs, "df should be in inputs")
        self.assertIn('df', outputs, "df should be in outputs for loc assignment")
        self.assertIn('idx', inputs, "idx should be in inputs")
        self.assertIn('value', inputs, "value should be in inputs")
    
    def test_attribute_assignment(self):
        """Test df.column = value is detected as modification"""
        code = "df.new_col = [1, 2, 3]"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        self.assertIn('df', inputs, "df should be in inputs")
        self.assertIn('df', outputs, "df should be in outputs for attribute assignment")
    
    def test_augmented_subscript_assignment(self):
        """Test df['column'] += 1 is detected as read and write"""
        code = "df['column'] += 1"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        self.assertIn('df', inputs, "df should be in inputs for augmented assignment")
        self.assertIn('df', outputs, "df should be in outputs for augmented assignment")
    
    def test_inplace_method_call(self):
        """Test df.sort_values(inplace=True) is detected as modification"""
        code = "df.sort_values(by='col', inplace=True)"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        self.assertIn('df', inputs, "df should be in inputs for inplace operation")
        self.assertIn('df', outputs, "df should be in outputs for inplace operation")
    
    def test_inplace_method_false(self):
        """Test df.sort_values(inplace=False) does NOT modify df"""
        code = "result = df.sort_values(by='col', inplace=False)"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        self.assertIn('df', inputs, "df should be in inputs")
        self.assertNotIn('df', outputs, "df should NOT be in outputs when inplace=False")
        self.assertIn('result', outputs, "result should be in outputs")
    
    def test_no_inplace_method(self):
        """Test df.sort_values() without inplace does NOT modify df"""
        code = "result = df.sort_values(by='col')"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        self.assertIn('df', inputs, "df should be in inputs")
        self.assertNotIn('df', outputs, "df should NOT be in outputs without inplace")
        self.assertIn('result', outputs, "result should be in outputs")
    
    def test_chained_subscript(self):
        """Test df[df['col'] > 0]['new_col'] = 1"""
        code = "df[df['col'] > 0]['new_col'] = 1"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        # df should be input (read for condition and assignment target)
        # and output (modified)
        self.assertIn('df', inputs, "df should be in inputs for chained operation")
        self.assertIn('df', outputs, "df should be in outputs for chained operation")
    
    def test_nested_subscript_assignment(self):
        """Test nested['level1']['level2'] = value"""
        code = "nested['level1']['level2'] = value"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        self.assertIn('nested', inputs, "nested should be in inputs")
        self.assertIn('nested', outputs, "nested should be in outputs")
        self.assertIn('value', inputs, "value should be in inputs")
    
    def test_define_then_modify(self):
        """Test that defining a variable then modifying it works correctly"""
        code = """df = pd.DataFrame()
df['col'] = [1, 2, 3]"""
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        # df is defined in block, so when we modify it, it should:
        # - NOT be in inputs (defined in block)
        # - BE in outputs (modified in block)
        self.assertIn('pd', inputs, "pd should be in inputs")
        self.assertNotIn('df', inputs, "df should NOT be in inputs (defined in block)")
        self.assertIn('df', outputs, "df should be in outputs")
    
    def test_regular_assignment_unchanged(self):
        """Test that regular assignments still work correctly"""
        code = "x = 1"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        self.assertEqual(len(inputs), 0, "No inputs expected")
        self.assertIn('x', outputs, "x should be in outputs")
    
    def test_multiple_modifications(self):
        """Test multiple different types of modifications"""
        code = """df['a'] = 1
df.b = 2
df.sort_values(inplace=True)
df.loc[0, 'c'] = 3"""
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        
        # All should recognize df as both input and output
        self.assertIn('df', inputs, "df should be in inputs")
        self.assertIn('df', outputs, "df should be in outputs")


if __name__ == '__main__':
    unittest.main()

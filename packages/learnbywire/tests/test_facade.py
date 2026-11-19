import importlib
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import learnbywire

class FacadeTests(unittest.TestCase):
    def test_lazy_import_does_not_load_training_dependencies(self):
        subprocess.run([sys.executable, "-c", "import sys,learnbywire; assert 'torch' not in sys.modules; assert 'optimizerplus' not in sys.modules; assert 'trainerplus' not in sys.modules"],check=True)
    def test_imported_classes_are_actual_component_classes(self):
        from learnbywire import TrainerPlus, OptimizerPlus
        import trainerplus, optimizerplus
        self.assertIs(TrainerPlus,trainerplus.TrainerPlus)
        self.assertIs(OptimizerPlus,optimizerplus.OptimizerPlus)
    def test_versions_distinguish_parent_and_children(self):
        self.assertEqual(learnbywire.__version__,'1.2.2')
        self.assertEqual(learnbywire.__trainer_version__,'1.1.1')
        self.assertEqual(learnbywire.__optimizer_version__,'1.2.2')
    def test_missing_child_is_actionable(self):
        with patch.object(learnbywire.importlib,'import_module',side_effect=ModuleNotFoundError('missing',name='optimizerplus')):
            with self.assertRaisesRegex(ModuleNotFoundError,'learnbywire-optimizerplus'):
                learnbywire.OptimizerPlus
    def test_nested_dependency_error_is_preserved(self):
        error=ModuleNotFoundError('dependency missing',name='nested_dependency')
        with patch.object(learnbywire.importlib,'import_module',side_effect=error):
            with self.assertRaises(ModuleNotFoundError) as caught:learnbywire.OptimizerPlus
            self.assertIs(caught.exception,error)
    def test_unknown_export_does_not_import(self):
        with patch.object(learnbywire.importlib,'import_module') as imported:
            with self.assertRaises(AttributeError):learnbywire.Unknown
            imported.assert_not_called()
    def test_introspection_lists_both_products(self):
        self.assertTrue({'TrainerPlus','OptimizerPlus','TrainingConfig'}.issubset(dir(learnbywire)))
if __name__=='__main__':unittest.main()

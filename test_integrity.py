"""Focused regression checks for interrupted training and invalid annotations."""
import bootstrap
import csv
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import experiment
from prepare_data import read_annotation


class IntegrityTests(unittest.TestCase):
    def make_run(self, directory, rows):
        root=Path(directory)
        checkpoint=root/'weights/best.pt';checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'file-presence-only-test')
        with (root/'results.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['epoch'])
            writer.writerows([[i] for i in range(rows)])
        return checkpoint

    def test_interrupted_training_is_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=self.make_run(directory,3)
            with self.assertRaisesRegex(RuntimeError,'Incomplete training'):
                experiment.completed_checkpoint(checkpoint,40)

    def test_complete_epoch_log_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=self.make_run(directory,40)
            self.assertTrue(experiment.completed_checkpoint(checkpoint,40))

    def test_missing_checkpoint_requests_training(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(experiment.completed_checkpoint(Path(directory)/'missing.pt',40))

    def test_invalid_annotation_is_rejected_without_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            image=Path(directory)/'crazing_test.jpg';xml=image.with_suffix('.xml')
            Image.new('L',(100,100),128).save(image)
            xml.write_text('<annotation><size><width>100</width><height>100</height></size>'
                           '<object><name>crazing</name><bndbox><xmin>1</xmin><ymin>1</ymin>'
                           '<xmax>101</xmax><ymax>90</ymax></bndbox></object></annotation>')
            with self.assertRaisesRegex(ValueError,'Invalid box'):
                read_annotation(xml,image)


def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IntegrityTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('Regression tests failed')
    # Invert the new completion guard in memory; the interruption test must fail.
    original=experiment.completed_checkpoint
    code=inspect.getsource(original).replace('if len(rows) != expected_epochs:',
                                            'if len(rows) == expected_epochs:')
    namespace=vars(experiment).copy()
    exec(compile(code,'<completion-guard-mutation>','exec'),namespace)
    try:
        experiment.completed_checkpoint=namespace['completed_checkpoint']
        output=io.StringIO()
        mutant=unittest.TextTestRunner(stream=output).run(
            IntegrityTests('test_interrupted_training_is_not_treated_as_complete'))
        detected=not mutant.wasSuccessful()
    finally:
        experiment.completed_checkpoint=original
    if not detected:raise RuntimeError('The tests did not catch the inverted completion guard')
    info={'regression_tests_passed':result.testsRun,'completion_guard_mutation_detected':detected,
          'scope':'Interrupted-training state and out-of-bounds annotation handling; numerical runtime covered by actual smoke/matrix/reload checks'}
    (bootstrap.ROOT/'results/code_checks.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('CODE_CHECKS_OK',json.dumps(info),flush=True)


if __name__=='__main__':
    main()

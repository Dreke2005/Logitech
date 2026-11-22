"""Keep unit tests away from the models: readings come from mocks, never from Ollama or the cloud."""
import paper_cloud
import paper_reader


def _no_model(*_args, **_kwargs):
    raise RuntimeError("Unit tests do not call the local models.")


paper_reader._ask = _no_model
paper_reader.installed_models = lambda: ("glm-ocr:q8_0",)


def _no_cloud(*_args, **_kwargs):
    raise paper_cloud.CloudUnavailable("Unit tests do not call the cloud.")


paper_cloud.MODE = "local"  # No test reaches Claude: cloud paths are tested with paper_cloud.ask mocked,
paper_cloud.ask = _no_cloud  # and a workbench that loads the user's "cloud" setting still stays local.

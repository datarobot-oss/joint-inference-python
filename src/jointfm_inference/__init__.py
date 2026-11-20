# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference runner that serves JointFM checkpoints over HTTP.

``LocalService`` starts a service from Python for the length of a ``with``
block, configured by ``ServiceSettings``; importing either loads neither torch
nor the model.
"""

from importlib.metadata import version

from jointfm_inference.local_service import LocalService, LocalServiceError
from jointfm_inference.service_settings import ServiceSettings

__version__ = version("jointfm-inference")

__all__ = ["LocalService", "LocalServiceError", "ServiceSettings", "__version__"]

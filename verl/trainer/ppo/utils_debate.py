# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Debate training: second actor/rollout uses ``Role.ActorRolloutB`` (see ``utils.Role``).

Python 3.12+ forbids subclassing an :class:`enum.Enum` to add members, so we extend :class:`Role` in
``utils.py`` instead of defining a ``DebateRole`` subclass.
"""

from verl.trainer.ppo.utils import Role

# Backward-compatible name for debate code paths.
DebateRole = Role

__all__ = ["DebateRole", "Role"]

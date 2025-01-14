# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Union, Any


@dataclass
class Example:
    x: Union[str, Any]
    y: Union[str, Any]

    def to_xml(self) -> str:
        return """<input>
{x}
</input>
<output>
{y}
</output>
""".format(
            x=self.x, y=self.y
        )

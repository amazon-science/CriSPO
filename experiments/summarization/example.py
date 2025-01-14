# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from crispo.task.example import Example


class SummarizationExample(Example):
    def to_xml(self):
        return """<input>
{article}
</input>
<summary>
{summary}
</summary>""".format(
            article=self.x, summary=self.y
        )

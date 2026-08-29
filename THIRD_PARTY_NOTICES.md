# Third-party notices

## TalkTuner

The synthetic conversations under `data/probe_training_datasets/` and the probe
training methodology are derived from
[TalkTuner-chatbot-llm-dashboard](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard).

MIT License

Copyright (c) 2024 Yida Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The local training implementation is model-agnostic and uses the selected model's
own chat template, hidden size, and number of decoder layers.

## SubPOP

The optional SubPOP experiment loads the gated `jjssuh/subpop` dataset at runtime.
SubPOP is released under CC BY-NC-SA 4.0 with additional access terms shown on its
Hugging Face dataset card. The repository does not include a copy of the dataset.

SubPOP is derived from Pew Research Center's American Trends Panel and the General
Social Survey 1972–2022 (Davern, Bautista, Freese, Herd, and Morgan; NORC, 2024,
2022 Release 3a).

The opinions expressed herein, including any implications for policy, are those
of the author and not of the survey research centers.

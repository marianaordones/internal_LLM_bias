# Probe training datasets

These compressed synthetic-conversation datasets were copied from the MIT-licensed
[TalkTuner repository](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard).
They are consumed directly as ZIP files by `training/train_demographic_probes.py`;
manual extraction is not required.

The class labels encoded in the filenames are:

- gender: `male`, `female`
- age: `child`, `adolescent`, `adult`, `older adult`
- education: `someschool`, `highschool`, `collegemore`
- socioeconomic status: `low`, `middle`, `high`

`llama_gender_3.zip` is retained as part of the upstream dataset collection, but
its entries use `_age_` in their filenames. The loader rejects those malformed
entries instead of silently assigning incorrect labels.

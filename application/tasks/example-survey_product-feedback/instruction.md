# Survey Product Feedback

We're gathering reactions to **FocusLoop**, a family coordination app concept.

Read the product brief, then complete every question in the questionnaire. We want to know how you'd price it, whether you'd try it, and how it would fit into household life — not feature ideas outside the brief.

## How to answer

- Read the brief before you start.
- Answer every required question.
- For multiple-choice, use the listed option ids.
- For rating scales, use a whole number in the given range.
- Give the answer alone unless a question also asks for a short reason or confidence.

## Output contract

The questionnaire (question ids, types, and option ids) is at
`/app/input/questionnaire.yaml`. Read it before answering — do not invent
question ids or option ids.

Write your final answers to `/app/output/survey_result.json` as JSON:

```json
{
  "answers": [
    {"questionId": "q0", "value": "q0_pay_when_roi_clear", "rationale": "optional short reason"}
  ],
  "trajectory": [
    {
      "timestamp": "2026-08-19T00:00:00Z",
      "actor": "persona",
      "action": "ask_question",
      "context": {"questionId": "q0", "questionType": "single_choice"},
      "outcome": {}
    }
  ]
}
```

- One `answers` entry per question id in `questionnaire.yaml`. `value` is the
  option `id` for `single_choice`, or an integer in `[minValue, maxValue]` for
  `likert`.
- One `trajectory` entry per question, `action` set to `"ask_question"`, with
  `context.questionId` and `context.questionType` matching the question.

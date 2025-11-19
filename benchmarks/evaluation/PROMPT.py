SYSTEM_PROMPT_IMAGE = """
You will be given:
- One or more page images (each page as a separate input image).
- A natural language question `q`.

Your task:
- Analyze the visual content carefully (e.g., text, patterns, symbols, diagrams, or objects) and determine the correct answer.

Output requirements:
- Include one concise `answer` field summarizing the final textual answer.

Return strictly in this JSON format:
{"answer": "<your concise answer here>"}

Do not include explanations, reasoning, or any other text outside the JSON.
"""

SYSTEM_PROMPT_IMAGE_BBOX = """
You will be given:
- One or more page images (each page as a separate input image).
- A natural language question `q`.

Your task:
1. Identify all regions (bounding boxes) on the given pages that contain the information necessary to answer the question.
2. Each bounding box must represent **one complete logical block** (a full paragraph, full table, or full chart) with **no truncation at the borders**.
3. Derive the **concise textual answer** to the question from the content inside those bounding boxes.

Bounding box format:
- Coordinates are absolute pixel integers from the top-left corner of each page.
- Format: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

Output requirements:
- You may return **single or multiple bounding boxes per page**, if relevant.
- Pages with no relevant content should have an empty list `[]`.
- Include one concise `answer` field summarizing the final textual answer.

Return strictly in this JSON format:
{
  "bboxes": [
    [
      [x1, y1, x2, y2],   // bounding boxes for page 1
      [x1, y1, x2, y2]
    ],
    [
      [x1, y1, x2, y2]    // bounding boxes for page 2 if exist
    ],
    ...
  ],
  "answer": "your concise final answer here"
}

Do not include explanations, reasoning, or any other text outside the JSON.
"""


SYSTEM_PROMPT_GEN_BBOX = """
You will be given:
- One or more page images (each page as a separate input image).
- A natural language question `q`.

Your task:
1. Identify all regions (bounding boxes) on the given pages that contain the information necessary to answer the question.
2. Each bounding box must represent **one complete logical block** (a full paragraph, full table, or full chart) with **no truncation at the borders**.

Bounding box format:
- Coordinates are absolute pixel integers from the top-left corner of each page.
- Format: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

Output requirements:
- You may return **single or multiple bounding boxes per page**, if relevant.
- Pages with no relevant content should have an empty list `[]`.

Return strictly in this JSON format:
{
  "bboxes": [
    [
      [x1, y1, x2, y2],   // bounding boxes for page 1
      [x1, y1, x2, y2]
    ],
    [
      [x1, y1, x2, y2]    // bounding boxes for page 2 if exist
    ],
    ...
  ]
}

Do not include explanations, reasoning, or any other text outside the JSON.
"""
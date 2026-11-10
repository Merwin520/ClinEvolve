from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.common import extract_jsons


class TaskAgent(AgentSystem):
    def forward(self, inputs):
        """
        Clinical Task Agent for MIMIC-CDM Experiment 1.

        Args:
            inputs (dict):
                Expected fields:
                    - domain: optional, default "mimic_cdm"
                    - hpi: history of present illness for current patient

                The patient identifier is controlled externally by
                agent.clinical_context and is never exposed to the model.

        Returns:
            tuple:
                - prediction (dict)
                - new_msg_history (list)
        """

        domain = inputs.get(
            "domain",
            "mimic_cdm",
        )

        hpi = str(
            inputs.get(
                "hpi",
                "",
            )
        ).strip()

        if not hpi:
            hpi = (
                "History of present illness is unavailable."
            )

        instruction = f"""
You are a clinical decision-making agent operating in the {domain} environment.

You are evaluating exactly one current patient.

The patient identity is controlled externally by the evaluation harness.
You must never request, infer, invent, or switch to another patient identifier.

The patient's HISTORY OF PRESENT ILLNESS is provided below.

====================
HISTORY OF PRESENT ILLNESS
====================

{hpi}

====================
END HISTORY
====================

You may obtain additional clinical evidence using the available tools.

The available clinical tools allow you to:

1. Obtain the physical examination.
2. Request specific laboratory tests.
3. Request imaging for a specific modality and anatomical region.

Important tool-use rules:

1. The history above is already available.
   Do NOT attempt to request the history again.

2. Laboratory testing must be requested specifically.
   For example:

   <json>
   {{
       "tool_name": "get_lab_results",
       "tool_input": {{
           "test_names": ["WBC", "Lipase"]
       }}
   }}
   </json>

   Do not assume that calling the laboratory tool gives you all laboratory
   information. Request only tests that are clinically relevant.

3. Imaging must be requested using both modality and anatomical region.
   For example:

   <json>
   {{
       "tool_name": "get_imaging_results",
       "tool_input": {{
           "modality": "CT",
           "region": "Abdomen"
       }}
   }}
   </json>

4. Use the physical examination tool when examination findings are necessary
   for clinical assessment.

5. Do not invent laboratory values, imaging findings, examination findings,
   diagnoses, treatments, or any other clinical facts that have not been
   provided.

6. Do not assume unavailable information is normal.

7. Gather evidence progressively.
   If the current evidence is insufficient for a clinically supported
   decision, request additional relevant evidence.

8. Avoid unnecessary or repeated tool calls.

9. Do not access or infer information from other patients.

10. Do not use cross-patient memory.

11. Do not modify your own code, prompts, tools, evaluator, or environment.

12. Do not claim certainty when evidence is insufficient.

13. When sufficient evidence has been collected, stop using tools and provide
    the final response.

Your final response MUST be returned in exactly one JSON block with this schema:

<json>
{{
    "response": {{
        "diagnosis": "...",
        "diagnostic_summary": "...",
        "treatment_plan": "...",
        "confidence": "...",
        "stop_reason": "..."
    }}
}}
</json>

Field requirements:

- "diagnosis":
  Your final diagnosis or most likely diagnosis based only on the evidence
  actually provided to you.

- "diagnostic_summary":
  A concise summary of the key clinical evidence supporting the diagnosis.
  Do not provide hidden chain-of-thought.
  Only provide a short evidence-based clinical summary.

- "treatment_plan":
  A concise management or treatment recommendation when appropriate.
  Base the recommendation only on available evidence.
  If evidence is insufficient, state that explicitly.

- "confidence":
  Must be exactly one of:
  "low", "moderate", "high".

- "stop_reason":
  Briefly explain why no more tool calls are required, for example:
  "sufficient evidence for diagnosis",
  "available evidence exhausted",
  or
  "insufficient evidence".

Do not output the final response JSON until you are finished using tools.
""".strip()

        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available=[
                "get_physical_exam",
                "get_lab_results",
                "get_imaging_results",
            ],
            multiple_tool_calls=False,
            max_tool_calls=40,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        prediction = {
            "diagnosis": "None",
            "diagnostic_summary": "",
            "treatment_plan": "",
            "confidence": "low",
            "stop_reason": (
                "unable to extract final response"
            ),
        }

        try:
            if not new_msg_history:
                self.log(
                    "Error: empty message history"
                )

                return (
                    prediction,
                    new_msg_history,
                )

            final_message = (
                new_msg_history[-1]
            )

            final_text = (
                final_message.get(
                    "text",
                    final_message.get(
                        "content",
                        "",
                    ),
                )
            )

            if final_text is None:
                final_text = ""

            final_text = str(
                final_text
            )

            extracted_jsons = (
                extract_jsons(
                    final_text
                )
            )

            if extracted_jsons:

                final_json = (
                    extracted_jsons[-1]
                )

                if (
                    isinstance(
                        final_json,
                        dict,
                    )
                    and "response"
                    in final_json
                ):

                    response = (
                        final_json[
                            "response"
                        ]
                    )

                    if isinstance(
                        response,
                        dict,
                    ):

                        confidence = str(
                            response.get(
                                "confidence",
                                "low",
                            )
                        ).strip().lower()

                        if confidence not in {
                            "low",
                            "moderate",
                            "high",
                        }:
                            confidence = "low"

                        prediction = {
                            "diagnosis":
                                response.get(
                                    "diagnosis",
                                    "None",
                                ),

                            "diagnostic_summary":
                                response.get(
                                    "diagnostic_summary",
                                    "",
                                ),

                            "treatment_plan":
                                response.get(
                                    "treatment_plan",
                                    "",
                                ),

                            "confidence":
                                confidence,

                            "stop_reason":
                                response.get(
                                    "stop_reason",
                                    "",
                                ),
                        }

                    else:
                        # Backward-compatible fallback.
                        prediction = {
                            "diagnosis":
                                str(response),

                            "diagnostic_summary":
                                "",

                            "treatment_plan":
                                "",

                            "confidence":
                                "low",

                            "stop_reason":
                                "legacy response format",
                        }

        except Exception as e:

            self.log(
                "Error extracting clinical "
                f"prediction: {e}"
            )

            prediction = {
                "diagnosis":
                    "None",

                "diagnostic_summary":
                    "",

                "treatment_plan":
                    "",

                "confidence":
                    "low",

                "stop_reason":
                    "prediction extraction error",
            }

        return (
            prediction,
            new_msg_history,
        )
system_prompts = {
    "Overview":'''1. Carefully read the provided consumer case file and think step-by-step to
extract a comprehensive overview. Start by identifying:
The product or service that is central to the grievance.
Next, describe the specific defect or issue the consumer experienced with it.
Then, consider what impact, harm, or inconvenience it caused the consumer.
Examine whether the consumer has tried any grievance mechanisms or escalation
steps (e.g., complaints, repairs, refund requests).
Analyze the response or counterclaims made by the opposite party or parties.
Clearly identify the parties involved in the case. If there are more than four
opposite parties, group or summarize them to maintain clarity.
Finally, reflect on the above details and summarize the core legal issue in
dispute in one sentence.
Now, write a single detailed overview paragraph (7–10 lines minimum)
incorporating all of the above points.
Format your response as:
Overview: [Write the full overview paragraph here]''',
"Sector":'''Carefully examine the provided case file and think step-by-step to identify the correct sector classification.
First, determine the product or service that is central to the grievance.
Then, analyze the identity or nature of the opposite party — what type of organization or business are they? (e.g., a bank, hospital, e-commerce site).
Use both the product/service and the opposite party’s nature to assess which sector best fits.
Refer to the list of sectors and select the single most appropriate match based on the combined information.
Do not explain your choice — only output the final classification in the required format.
Your response must strictly follow this format (no extra text or explanation):
Sector:- [Sector Name], [Sector Code]
Use only one of the following predefined sectors:
Banking and Financial Services  101  
Insurance   102  
Retail - Clothing   103  
Retail - Electronics    104  
Retail - Home & Furniture   105  
Retail - Groceries and FMCG 106  
Retail - Beauty & Personal Care 107  
E-commerce  108  
Telecommunications  109  
Consumer Electronics 110  
Healthcare and Pharmaceuticals 111  
Medical Services (including Negligence) 112  
Transport - Airlines    113  
Transport - Railways    114  
Real Estate 115  
Utilities (Electricity, Water)  116  
Automobiles 117  
Food Services   118  
Travel and Tourism 119  
Education   120  
Entertainment and Media 121  
Legal Services  122  
Home Services   123  
Sports and Recreation   124  
Technology Services 125  
Legal Metrology 126  
Petroleum   127  
Postal and Courier  128  
Others  999
''',
"Issues":'''3. Issues:- Carefully read the case file and follow these reasoning steps to
extract the key legal issues in dispute:
Identify the claims made by the complainant—what specific allegations,
factual assertions, or complaints have they raised?
Next, analyze the responses or counterclaims made by the opposite party—what
parts of the complainant’s case do they deny, reject, or challenge?
For each area of disagreement, formulate a precise, specific issue that reflects
a point of contention between the parties.
Make sure each issue captures only one distinct claim or factual dispute.
Exclude any background details, narrative summaries, or uncontested facts.
Present your final answer in this strict format:
Issues:-
1) [First issue]
2) [Second issue]
...(and so on)
Only include actively disputed issues that form part of the legal conflict.
''',
"Evidence by Complainant":'''4. Evidence by the complainant: Carefully examine the complaint copy filed
by the complainant and follow these steps to extract the evidentiary items:
Scan through the text to identify any explicit references to physical or
digital materials submitted as part of the complaint.
Look for items such as receipts, invoices, tickets, contracts, bills, emails,
letters, photographs, videos, or any other documents cited by the complainant.
Ensure that each item is mentioned in the complaint itself and is part of the
official submission before the court.
For each evidence item, write a brief but clear description, focusing only on
its type and relevance.
Do not include items implied but not mentioned, or any interpretation,
background, or legal commentary.
Output your answer strictly in this format:
Evidence presented by the complainant:-
CE1. [Brief description of the first evidence item]
CE2. [Brief description of the second evidence item]
CE3. [Brief description of the third evidence item]
(...continue as needed)
Do not include anything outside this format.''',
"Evidence by Opposing Party":'''5. Evidence by the opposite party: Carefully read the written statement or reply
filed by the opposite party in the case file and follow these
steps to extract the evidence they have presented:
Identify all explicitly mentioned documents or materials submitted by the
opposite party as part of their defense or response.
Look for references to bills, receipts, contracts, photographs, videos,
official records, letters, emails, or any other material intended to support
their version of events. Verify that each item is specifically mentioned in the
written statement or attached as supporting material by the opposite party.
For each valid evidence item, write a concise and factual description,
limited to what is stated in the file.
Do not include any inferred evidence, commentary, or background explanation.
Output your answer strictly in the following format:
Evidence presented by the opposite party:-
OPE1. [Brief description of the first evidence item]
OPE2. [Brief description of the second evidence item]
OPE3. [Brief description of the third evidence item]
(...continue as needed)
Do not include anything beyond the list. No summaries, no headings, no
reasoning—just the formatted output.
''',
"Reliefs":'''6. Relief:- Follow these reasoning steps to extract the reliefs requested:
1. Locate the prayer or relief section of the complaint, usually found at the
end of the complaint copy.
2. Identify each specific request made by the complainant to the court — this
could include refunds, compensation, damages, interest, litigation costs,
or any declaratory or injunctive relief.
3. Ensure that each relief is explicitly mentioned in the prayer and not
inferred from the narrative.
4. If a monetary amount is stated, include the figure as written.
5. List each relief as a separate bullet point, without interpretation, summary,
or rephrasing.
Present your final answer strictly in this format:
Reliefs:-
[First relief requested, include figures if mentioned]
[Second relief requested] [Third relief requested](...and so on)
Do not include any additional explanation, headings, or commentary—only the
relief list.
'''
}
# system_prompts = {
#     "Overview": "Extract a concise, factual overview (5-6 sentences) from the following consumer case file.",
#     "Sector": "From the following consumer case, identify the relevant consumer sector (e.g., Banking, Insurance, Retail, etc.). Output only the sector.",
#     "Issues": "List the main issues in dispute between the parties in the following consumer case.",
#     "Evidence by Complainant": "Summarize the evidence presented by the complainant in the following consumer case.",
#     "Evidence by Opposing Party": "Summarize the evidence presented by the opposing party in the following consumer case.",
#     "Reliefs": "State the reliefs claimed and/or granted in the following consumer case."
# }

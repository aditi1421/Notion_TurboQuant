"""
Process fetched Notion pages into chunks.
This script contains the actual page data pulled from Aditi's Notion workspace.
"""

import json
import re
import os

def clean_text(text):
    """Strip Notion markup artifacts, images, empty blocks."""
    # Remove image URLs
    text = re.sub(r'!\[.*?\]\(https://prod-files-secure.*?\)', '', text)
    # Remove file/attachment refs
    text = re.sub(r'<file src=".*?"></file>', '', text)
    # Remove page/ancestor XML tags
    text = re.sub(r'<(?:page|ancestor-path|parent-page|ancestor-\d+-page|properties|content|empty-block/).*?>', '', text)
    text = re.sub(r'</(?:page|ancestor-path|content|properties)>', '', text)
    # Remove table HTML-like markup but keep content
    text = re.sub(r'</?(?:table|tr|td|th|colgroup|col)[^>]*>', ' ', text)
    # Remove remaining HTML-ish tags
    text = re.sub(r'<[^>]+/?>', ' ', text)
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into chunks by words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# All pages fetched from Notion workspace
PAGES = [
    {"id": "2eec57e7-772a-80cc-92de-c13b93a628ed", "title": "Workflow Engine", "content": """Workflow such that there is multiple actions: Drafting, Commercial Contracts. Choose contract, Have conversation. Contracts & different types of contracts. Request, drafted, org review, counsel review. External counsel should be able to look and review. Our counsel, Our lawyers, External counsel, External lawyers. Aadhar e-signing. How can we ask more pointed qs without having seeded data? Getting information from the user. Agentify it. Blackboard: Conversation - Blackboard (Info from draft & describes open-ended draft). Create Contract: name, type, counterparty, category. Templates & Conversation. Upload existing ref contracts. NYA-84: Unified Workflow Engine (workflows table, status machine, RBAC). FUNDRAISING, REVIEW, EMPLOYMENT, GOVERNANCE, POLICIES. Conversation-first pattern used by Fundraising, Governance: Open-ended AI Q&A, Extract context, Spine, Flesh out. Template-based pattern used by Employment, Policies: Close-ended Q&A, Variable infill, Review. All documents flow through: CREATED, DRAFTED, INTERNAL_ITERATION, EXTERNAL_ITERATION, SIGNED. With role-gating at each stage (Client, Internal Counsel, Counterparty). A workflow is a wrapper around the existing draft + agent session. It adds: Status tracking, Role gating, Counterparty loop. Roles: CLIENT (Startup founder), CLIENT_COUNSEL (external lawyer), INTERNAL_COUNSEL (platform lawyer), COUNTERPARTY (Investor, vendor), COUNTERPARTY_COUNSEL (Their lawyer)."""},

    {"id": "2c4c57e7-772a-80b7-aa57-e8ee67562153", "title": "Document Processing with Langchain", "content": """Understand langchain for upload, segmentation, ocr extraction, summary and metadata. Upload pipeline: Storage (PDF file to Blob URL), Segment (Google Vision crop OCR + GPT-5, split PDFs + labels), OCR (Google Vision API to ocr_text.txt, structured_ocr.json), Metadata (GPT-4o to metadata.json), Summary (GPT-4o-mini to summary.txt). Have Prompt Templates with langchain so future changes are easy to maintain. Chat bot which interacts with user, goes back to the pdf and then gives a final draft. The PDF is processed on upload into OCR, metadata, chunks, and embeddings. The chatbot retrieves relevant segments during conversation and reasons using the LLM. Sessions are linked: case_id, agent_session, messages, draft. For langchain, its 3 chains: Upload flow, Chatbot Retrieval, Final Draft generation. RAG pipeline: Question, Find relevant pieces of documents, Feed only those pieces + question to LLM, Answer. DocumentDigest, GapDetectionRunnable, QuestionAskingRunnable, UserAnswer, StateUpdateRunnable, Loop, DraftGenerationRunnable. Instead of pushing all the documents into the LLM we only push the small parts that are relevant to the user's question."""},

    {"id": "32ec57e7-772a-80f2-bc17-f508aaaf88eb", "title": "My Notes on Engineering", "content": """When I started out as a junior dev, I pulled long hours so I could deliver on time. I always felt that failing to deliver on time was never an option. As soon as I understand what the work is, I break it all down, ideally on a whiteboard or paper. An approach that consistently worked for me is approaching problems from the customer's perspective, and being genuinely curious. Task breakdown: early in my career, there was a senior engineer who was methodical about breaking down tasks and making estimates, even for seemingly trivial projects and it worked! If I no longer get energy from the work I do, then I basically stop enjoying it and this can be a nudge to start to look for something else. I learned a lot about ownership, the importance of an eye for detail, and collaborating with others."""},

    {"id": "304c57e7-772a-8018-8475-ce9fde2ec21d", "title": "Startup Legal Platform", "content": """AI-guided document drafting and AI-powered redline review, with a qualified advocate in the loop on every document. Two workflows: founder describes what they need, AI drafts, counsel reviews, client gets document. Or: founder uploads an investor document, AI redlines it, founder understands what they're signing. Market: 16,000+ startups in Bangalore alone, 47% of India's startup funding flows through the city, seed stage funding up 26% YoY. Indian legal AI is growing at 23% CAGR but still early and underpenetrated. Nobody has built this specific product for this specific customer, the pre-seed to Series B founder without in-house counsel. Your document, reviewed by a real lawyer, in 24 hours. A conservative estimate for Year 1 is 10 documents across 4-5 document types. DocuSign integration with signing workflow: Initiate Signing, Signer Gets Embedded URL, Signer Signs in DocuSign, DocuSign Fires Webhooks, Frontend Polls for Status. Workflow: FINALIZED, SIGNING_IN_PROGRESS, SIGNED. Sequential Signing: Signers go in order via routing_order."""},

    {"id": "2e6c57e7-772a-80c3-8c8c-c9a716bddec7", "title": "File Request Endpoints", "content": """File Request Workflow - Frontend Integration. Create a file request (Requester to Requestee): POST /v1/file-requests. List my requests: GET /v1/file-requests/me. Upload file (Requestee): Get upload URL, Upload to Azure, Confirm upload. Approve/Reject file (Requester). Request Status Flow: PENDING, UPLOADED, APPROVED/REJECTED. Permissions: Any org user can create a file request, Only requestee can upload files, Only requester can approve/reject files. File Visibility Rules: Files with filetype PENDING_APPROVAL are completely hidden from case files list. Multiple Files Per Request: Requestee can upload multiple files, Each file gets uploaded separately, Requester approves/rejects each file individually."""},

    {"id": "2ffc57e7-772a-80e0-9b8d-e914db9d8a72", "title": "Review Flow & Checklists", "content": """When a user creates an org, they select what they want to do: fundraising, employment etc. They have a list of check items to upload like SEBI doc, 3/4 items. Once they click on it, they need to upload the doc and then they can view/delete/add docs based on the checklist. Checklist items is the master list of what documents are required per type. It's the template. For FUNDRAISING: SEBI Registration Certificate, Term Sheet, Board Resolution. Seed data (global templates): checklist_types (FUNDRAISING, EMPLOYMENT, COMPLIANCE), item_types (specific documents per type). Per-org data: checklists (org instance), items (uploaded files). Deploy: Seed checklist_items with items per bucket. User opens review: GET checklist_items WHERE contract_bucket = contract.bucket."""},

    {"id": "32cc57e7-772a-800f-a2e4-de2fa2df3739", "title": "LLM-RL Learning Path & AI Agent Tips", "content": """If you want to learn LLM-RL: go through David Silver's playlist with Sutton's book, go through the policy grad blog by Karpathy, try to formulate the MDP for applying RL on LLMs without any external help, implement PPO via PyTorch and play with hyper params. Experiment a lot, Think like AI native people. Make your instruction as clear as possible and provide a lot of details. Provide enough context that's relevant to the task at hand so your agent doesn't have to guess. Keep your context as short as possible so that it's able to better focus on what's important. Having a separate document for each project lets you work on separate projects at the same time. Keep each session as short as possible. Ask the AI agent to summarize progress in project-progress.md."""},

    {"id": "302c57e7-772a-8061-bd70-d22c97bc6b16", "title": "Draft Types & Annexures", "content": """Per draft type, All possible annexures. For each case, we have non-action and action metrics. For one button, click view and it generates a list of notices. How much money will you recover, How much is remaining. Index/table of contents on top. Calculate Annexures, calculate Page numbers. Ingest loan documents, save it. Loan becomes non performing asset after 90 days, then its past due."""},

    {"id": "2eac57e7-772a-803c-9822-f3421cac8537", "title": "Website Design Thoughts", "content": """The core issue: Our website looks like an AI-generated template with no soul, no proof, and no people. No credibility signals: No investor logos, no client logos, no testimonials, no team photos. No human element: Zero faces, zero names, zero story. Generic visual identity: Washed-out blue palette looks like every other SaaS template. Unclear positioning: Legal OS is vague, competitors own clear categories like Justice Tech Stack, AI Paralegal. Color scheme: Near-black (#121212) + Electric teal (#14b8a6) + White. Tagline: Legal Infrastructure for Indian Enterprise. Mission Line: Because when business scales, legal shouldn't be the thing that breaks. Solution Section: Contracts (Extract obligations, risks, deviations automatically), Obligations (Track every deadline), Litigations (Structure notices, claims, disputes), Evidence (Centralize documents, research, audit trails)."""},

    {"id": "2d3c57e7-772a-80ca-ba37-d2edca58d1ff", "title": "Translation Service Endpoints", "content": """Workflow Sequence: Create order, Upload files, Set target pages, Trigger translation, Assign proofreader, Proofreader edits via Google Doc, Mark proofreading complete, Approve QC, Calculate total, Create payment, Download final. POST /v1/translation/orders with source_language and target_language. Upload file to order. Set target pages (use all for all pages). Trigger Auto Machine Translation returns job_id. Get Google Doc URL for proofreader. Assign proofreader by email (shares Google Doc). Mark PROOFREAD status. Approve QC (automatically exports Google Doc to DOCX). Calculate order total based on pages times per_page_cost. PhonePe payment integration: initiate-payment returns payment_url, Razorpay webhook for payment confirmation."""},

    {"id": "2e7c57e7-772a-8029-84f3-c0a7c48cdd85", "title": "Startup Motion Understanding", "content": """Startup Motion is a service that connects startups to lawyers. Two sides: external and internal. External has Six buckets: Corporate Documents (IP bundled), Commercial contracts, Employment and people, Fundraising, Compliance and Policies, Disputes. Dashboard has a Snapshot of all four buckets with No. of documents, Notifications and reminders. Create action tab: Action type (Draft/Review & Negotiate/Assign dispute), Category, Status, Last updated. For internals: Admin counsel can assign tasks to a counsel internally. Just two buckets for the counsel: documentation and disputes. Payment: Credit based system. Reload credits using payment."""},

    {"id": "2c3c57e7-772a-80e5-8b60-ec3355989504", "title": "Langchain vs LlamaIndex", "content": """LangChain is an open-source framework that simplifies building applications powered by Large Language Models like GPT-4 or Gemini. Without LangChain you write raw API calls, manually handle prompts, context, memory, documents, vector DBs. With LangChain you get reusable components for all of this. LlamaIndex will be good ONLY for one thing: building a clean RAG pipeline. We shouldn't use LlamaIndex for the whole LLM architecture, but it can improve our RAG layer: document ingestion, chunking, embeddings, and retrieval. RAG becomes a core system because: You have many documents in each case, The LLM should not hallucinate, Drafting should use actual evidence from files, You need traceability."""},

    {"id": "2d3c57e7-772a-8042-a3dd-d7d5893ba914", "title": "OCR Pipeline", "content": """High-Level Flow: PDF File, Convert to Images, Send to Google Vision API, Get Text Back. PDF to Images: Converts each page into a separate image using 150 DPI resolution. Batch Processing: Processes 10 pages at a time in batches instead of sending all pages at once which could timeout or overload the API. Google Vision API Call: Uses DOCUMENT_TEXT_DETECTION mode optimized for documents/printed text. Collect Results: Stores text for each page with its page number. Aggregate Text: Joins all page texts together with double newlines."""},

    {"id": "2e8c57e7-772a-8093-bd83-ed8d1f5fdfc6", "title": "E-Courts Translation Full Flow", "content": """Court system with 25 High Courts, language mapping, court_id for isolation. Models: court, user, judgement (workflow states), sc_judgement (assignment logs). Routers: admin (filters by court_id), superadmin (distribution algorithm). User logs in, auth validates credentials, returns JWT token. Admin uploads judgement stored in judgement table with court_id. Admin assigns to proofreader, checks court_id match. Proofreader completes work, uploads document, updates timestamps. Invoice generated, calculates pages, creates payment record."""},

    {"id": "2e3c57e7-772a-80dc-ae54-ce4ff85bd984", "title": "Notebook LM Architecture", "content": """NotebookLM is an AI-first notebook powered by Google's Gemini 1.5 Pro model. Its primary differentiator is grounding: restricts LLM responses to uploaded documents to reduce hallucination. Ingestion Layer: Up to 50 sources per notebook, 25 million words total. Supports Google Docs, Slides, PDFs, text files, web URLs. Embedding Generation: Text chunk turned into dense vector representation, captures semantic meaning. Google's Vertex AI text embeddings API with gemini-embedding-001. Chunking Strategies: Recursive character text splitting, Markdown header splitting, Layout Aware document chunking. Query Processing: Semantic Search using dense vector embeddings, cosine/dot/euclidean distance. Hybrid Search combining keyword and semantic approach. Relevance Ranking with cross-encoders for re-ranking."""},

    {"id": "2d1c57e7-772a-8071-bea5-dc93318a715d", "title": "Google Docs API", "content": """Google Docs API cannot import/convert files. It can only: Create blank documents, Read existing document content, Modify content (insert text, apply formatting). Google Drive API is required to: Upload a DOCX file, Convert it to Google Docs format during upload, Set sharing permissions. PDF to MT to Docx pipeline."""},
]


def main():
    os.makedirs("data", exist_ok=True)

    all_chunks = []
    for page in PAGES:
        cleaned = clean_text(page["content"])
        chunks = chunk_text(cleaned, chunk_size=400, overlap=50)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "page_id": page["id"],
                "title": page["title"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk,
            })

    output_path = os.path.join("data", "notion_chunks.jsonl")
    with open(output_path, 'w') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')

    print(f"Processed {len(PAGES)} pages into {len(all_chunks)} chunks")
    print(f"Saved to {output_path}")

    # Also save test queries derived from page titles + some content queries
    queries = [p["title"] for p in PAGES]
    queries += [
        "How does the RAG pipeline work?",
        "What is the workflow state machine?",
        "How does DocuSign signing integration work?",
        "What embedding models does NotebookLM use?",
        "How does the translation service handle payments?",
        "What are the file request API endpoints?",
        "How does OCR extraction work on PDF documents?",
        "What is the difference between Langchain and LlamaIndex?",
        "How does the checklist review flow work?",
        "What is Startup Motion and how does it connect startups to lawyers?",
    ]
    queries_path = os.path.join("data", "test_queries.json")
    with open(queries_path, 'w') as f:
        json.dump(queries, f, indent=2)
    print(f"Saved {len(queries)} test queries to {queries_path}")


if __name__ == "__main__":
    main()

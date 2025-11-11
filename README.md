# Brampton HR "AI Mentor" Chatbot (PoC)

![Status: Prototype](httpsa://img.shields.io/badge/Status-Prototype-blue.svg)

This repository contains a fully functional, AI-powered chatbot prototype built as part of a Proof of Concept (PoC) proposal for the **City of Brampton's AI PoC Program**.

The chatbot, named the "AI Mentor," is a **Retrieval-Augmented Generation (RAG)** application. It is designed to provide safe, accurate, and sourced answers to employee questions by using a secure knowledge base of official HR policy documents.

**This prototype solves two of the city's stated needs:**
1.  **Project #7: Human Resources Self Service** (by answering policy questions)
2.  **Project #6: Human Resources L&D** (by proactively nudging for compliance training)

---

## 🚀 Live Demo

A live version of this prototype, deployed via Streamlit Community Cloud, can be accessed here:

**[https://bramptonhrpoc.streamlit.app](https://bramptonhrpoc.streamlit.app)**

*(Note: The live demo requires a `GEMINI_API_KEY` to be set in the Streamlit Cloud secrets.)*

---

## ✨ Key Features

### 1. Secure RAG Knowledge Base
The AI Mentor is designed with a "Protection by Design" framework. It cannot "hallucinate" or provide outside information. Its knowledge is strictly limited to the official documents ingested into its vector store:
* `HRM-100: Employee Code of Conduct`
* `HRM-150: Respectful Workplace Policy`
* `HRM-160: Recruiting and Retaining Top Talent`
* `HRM-210: Salary Administration Policy`
* `ANI-140: Anti-Racism and Inclusion Policy`
* `Employee Code of Conduct Handbook`
* GBA+ Mandate & Course Info

### 2. The "Golden Thread": Proactive L&D Nudging
This is the prototype's core strategic feature. The RAG prompt is engineered to find a specific connection ("Golden Thread") between policies.

**Demo Scenario:**
1.  A manager asks a simple HR question: `"How do I start the hiring process?"`
2.  The AI answers this query using the `HRM-160` (Recruiting) policy.
3.  It then **proactively** adds: `"Additionally, please be aware that Policy HRM-160 requires all hiring committee members to complete the 'Mandatory Recruitment and Diversity Learning Series.' You can find this module on the L&D portal."`

This single feature proves the app is an "L&D Co-Pilot," not just a simple chatbot.

---

## 🛠 Tech Stack

* **App Framework:** Streamlit
* **AI Orchestration:** LangChain
* **LLM:** Google Gemini (`gemini-2.5-pro`)
* **Embeddings:** Google Gemini (`text

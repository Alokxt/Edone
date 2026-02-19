# Edone – AI Powered Education Platform (Backend)

Edone is an AI-powered education backend designed with structured AI workflows, cost optimization, and real-world deployment constraints in mind.

It combines Retrieval-Augmented Generation (RAG), semantic caching, and Redis-backed state machines to power intelligent learning features.

---

## 🌐 Live Demo
**Live Link:**  
(Add your Render URL here)

## 📖 Architecture & Design Breakdown
**Medium Article:**  
https://medium.com/p/9a53c6e88573?postPublishedType=initial

---

## 🚀 Core Features

- **Syllabus-Constrained RAG Chatbot**  
  Topic-level embeddings for boundary-controlled AI responses.

- **Two-Layer Redis Caching**  
  - Exact match caching  
  - Semantic similarity caching  
  → Reduces latency and LLM cost.

- **Mock Interview Simulator**  
  - Redis-backed stateful session management  
  - Finite state transitions (`start → attempted → passed`)  
  - LLM-based scoring and structured insights  

- **AI Quiz Generation**  
  - Weekly aptitude quiz  
  - Context-aware syllabus-based quiz generation  

---

## 🏗 System Highlights

- Retrieval-Augmented Generation (RAG)
- Dynamic state transitions over a single API
- Lazy loading for memory optimization (512MB free-tier constraint)
- Cost-aware LLM usage
- TTL-based session lifecycle cleanup

---

## 🛠 Tech Stack

- Python
- Django / Django REST Framework
- Redis
- Vector Database
- Langchain
- LLM API Integration
- Deployed on Render (Free Tier)

---

## 📌 Note

This repository focuses on backend AI orchestration and system design.  
For detailed architecture decisions, trade-offs, and design challenges, refer to the Medium article linked above.

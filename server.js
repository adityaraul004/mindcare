// server.js
import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import { Configuration, OpenAIApi } from "openai";
import { initializeApp, applicationDefault } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import dotenv from "dotenv";

dotenv.config();

// Init Firebase
initializeApp({
  credential: applicationDefault(),
});
const db = getFirestore();

// Init Express
const app = express();
app.use(cors());
app.use(express.json());

// Init OpenAI
const openai = new OpenAIApi(
  new Configuration({ apiKey: process.env.OPENAI_API_KEY })
);

// HuggingFace Sentiment API
const HF_MODEL =
  "j-hartmann/emotion-english-distilroberta-base";
const HF_API_URL = `https://api-inference.huggingface.co/models/${HF_MODEL}`;
const HF_API_KEY = process.env.HF_API_KEY;

// High-risk keywords
const HIGH_RISK_KEYWORDS = [
  "kill myself",
  "end my life",
  "i want to die",
  "suicide",
  "self harm",
];

// Sentiment Detection
async function detectSentiment(text) {
  const response = await fetch(HF_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ inputs: text }),
  });
  const result = await response.json();
  return result[0]?.[0]?.label || "neutral";
}

// Risk Detection
function isHighRisk(text) {
  const lower = text.toLowerCase();
  return HIGH_RISK_KEYWORDS.some((kw) => lower.includes(kw));
}

// Chat Endpoint
app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: "Message is required" });

    // Sentiment Analysis
    const sentiment = await detectSentiment(message);

    // Risk Detection
    const risk = isHighRisk(message);

    // OpenAI Chat
    const completion = await openai.createChatCompletion({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "You are an empathetic mental health support chatbot. Always respond kindly, encourage conversation, and avoid giving medical advice. If user is at risk, encourage professional help.",
        },
        { role: "user", content: message },
      ],
    });

    const reply =
      completion.data.choices[0]?.message?.content ||
      "I'm here to listen. Can you tell me more about how you're feeling?";

    // Save to Firestore
    await db.collection("chats").add({
      message,
      reply,
      sentiment,
      risk,
      timestamp: new Date(),
    });

    // Save mood
    await db.collection("moods").add({
      mood: sentiment,
      timestamp: new Date(),
    });

    res.json({ reply, sentiment, risk });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Chat processing failed" });
  }
});

// Get Mood Log
app.get("/mood", async (req, res) => {
  try {
    const snapshot = await db
      .collection("moods")
      .orderBy("timestamp", "desc")
      .get();
    const moods = snapshot.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    }));
    res.json(moods);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch moods" });
  }
});

// Post Mood
app.post("/mood", async (req, res) => {
  try {
    const { mood } = req.body;
    if (!mood) return res.status(400).json({ error: "Mood is required" });

    await db.collection("moods").add({
      mood,
      timestamp: new Date(),
    });

    const snapshot = await db
      .collection("moods")
      .orderBy("timestamp", "desc")
      .get();
    const moods = snapshot.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    }));

    res.json(moods);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to save mood" });
  }
});

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

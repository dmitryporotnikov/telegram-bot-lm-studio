# Simple telegram bot to talk to LM Studio over OpenAI API

This repository contains a Telegram bot that interacts with a local language model via LM Studio. The bot maintains contextual conversations, trims chat history to stay within token limits, and provides concise AI-generated responses. 

The bot assumes LM Studio’s OpenAI-compatible API is accessible at http://IP:1234/v1. Token counting follows an approximation method optimized for ChatML formatting. Ensure your model supports similar token limits as configured. 

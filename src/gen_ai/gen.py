import os
from google import genai 
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from openai import OpenAI

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")   
gemini_client = genai.Client(api_key=gemini_api_key)

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

def build_prompt (context,question):
    return f"""Use the context below to answer the user's question. The context comes from a PDF that the user is reading.

Treat the provided PDF context as the primary source of truth.

Follow these rules:

If the answer is directly supported by the PDF context, answer using the information from the PDF.
If the question is related to the PDF but requires a basic explanation, definition, example, or connection that is not explicitly stated in the PDF, you may use your general knowledge to provide a helpful explanation. Clearly distinguish this additional information from what is stated in the PDF.
Do not contradict, modify, or replace information provided by the PDF with your general knowledge.
If the PDF provides only part of the answer, use the PDF information first and supplement it with relevant general knowledge when necessary.
If the question is completely unrelated to the PDF, answer it using general knowledge, but make it clear that the answer is not based on the PDF.
Do not invent information or claim that something is stated in the PDF when it is not.
Keep the answer relevant to the user's question and the PDF's context.

    Context : {context}

    Question: {question}
    Answer:"""

def generate_answer(context, question, stream=False):
    """
    stream=False (default, unchanged behavior): returns the full answer string.
    stream=True: returns a generator yielding text chunks as they arrive,
    for use with st.empty()-style progressive rendering in the UI.
    """
    prompt = build_prompt(context, question)
 
    if stream:
        return _generate_answer_stream(prompt)
    return _generate_Answer_blocking(prompt)

def _generate_Answer_blocking(prompt):
    try:

        # GEMINI main
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text

    except Exception as exp:

        print(f"[Gemini failed: {exp}] Falling back to OpenAI...")

        # GROQ FALLBACK
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages = [{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

def _generate_answer_stream(prompt):
    try:
        # GEMINI main (streaming)
        stream = gemini_client.models.generate_content_stream(
            model="gemini-3.6-flash",
            contents=prompt
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
 
    except Exception as exp:
        print(f"[Gemini streaming failed: {exp}] Falling back to Groq...")
 
        groq_stream = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in groq_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

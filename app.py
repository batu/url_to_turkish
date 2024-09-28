import os
import logging
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, session
from youtube_transcript_api import YouTubeTranscriptApi
import io
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from urllib.parse import urlparse, parse_qs
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Or use a fixed secret key

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
LANGSMITH_ENDPOINT = os.getenv('LANGSMITH_ENDPOINT')

# Ensure that API keys are set
if not OPENAI_API_KEY or not LANGSMITH_API_KEY:
    raise ValueError(
        "Missing API keys. Please set OPENAI_API_KEY and LANGSMITH_API_KEY in your .env file."
    )

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks import tracing_v2_enabled

# Configure LangChain
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = LANGSMITH_ENDPOINT
os.environ['LANGCHAIN_API_KEY'] = LANGSMITH_API_KEY

proxy_username = os.getenv('PROXY_USERNAME')
proxy_password = os.getenv('PROXY_PASSWORD')
proxy_port = 8080
proxy_host = "gate.nodemaven.com"
proxy_url = f'http://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}'


def extract_video_id(youtube_url):
    """Extract the video ID from the YouTube URL."""
    try:
        url_data = urlparse(youtube_url)
        query = parse_qs(url_data.query)
        if 'v' in query:
            return query['v'][0]
        elif 'youtu.be' in youtube_url:
            return url_data.path[1:]
        else:
            return None
    except Exception as e:
        logging.error(f"Error extracting video ID: {e}")
        return None


def get_transcript(video_id):
    """Fetch the transcript for a given YouTube video ID."""
    try:
        logging.info("Fetching transcript...")
        proxy = {'http': proxy_url, 'https': proxy_url}
        if not os.getenv('USE_PROXY'):
            proxy = None
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id,
                                                                proxies=proxy)
        transcript = transcript_list.find_transcript(
            ['en', 'en-GB', 'en-US', 'de', 'fr']).fetch()
        full_transcript = ' '.join([entry['text'] for entry in transcript])
        logging.info("Transcript fetched successfully.")
        return full_transcript
    except Exception as e:
        logging.error(f"Error fetching transcript: {e}")
        return None


def split_text(text, max_tokens):
    """Split text into chunks that fit within the max token limit."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    max_words = max_tokens * 0.75  # Approximate words per token

    for word in words:
        current_chunk.append(word)
        current_tokens += 1
        if current_tokens >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_tokens = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def repair_transcript(transcript_text, llm):
    """Use OpenAI's API via LangChain to correct transcription errors."""
    max_tokens = 2048  # Adjust based on model's context window
    transcript_chunks = split_text(transcript_text, max_tokens)

    repaired_chunks = []
    for i, chunk in enumerate(transcript_chunks):
        logging.info(f"Repairing chunk {i+1}/{len(transcript_chunks)}...")
        messages = [
            HumanMessage(
                content=
                f"Please correct any transcription errors in the following text:\n\n{chunk}"
            )
        ]
        response = llm.invoke(messages)
        repaired_text = response.content
        repaired_chunks.append(repaired_text.strip())
    repaired_transcript = ' '.join(repaired_chunks)
    logging.info("Transcript repaired successfully.")
    return repaired_transcript


def translate_to_turkish(text, llm):
    """Use OpenAI's API via LangChain to translate text into Turkish."""
    max_tokens = 2048  # Adjust based on model's context window
    text_chunks = split_text(text, max_tokens)

    translated_chunks = []
    for i, chunk in enumerate(text_chunks):
        logging.info(f"Translating chunk {i+1}/{len(text_chunks)}...")
        messages = [
            HumanMessage(
                content=
                f"Please translate the following text into Turkish:\n\n{chunk}"
            )
        ]
        response = llm.invoke(messages)
        translated_text = response.content
        translated_chunks.append(translated_text.strip())
    translated_text = ' '.join(translated_chunks)
    logging.info("Translation completed successfully.")
    return translated_text


def generate_pdf(text):
    """Generate a well-formatted PDF file from the given text using Platypus and DejaVu Sans."""
    buffer = io.BytesIO()

    # Register DejaVu Sans font
    font_path = os.path.join('static', 'fonts', 'DejaVuSans.ttf')
    if not os.path.exists(font_path):
        logging.error(f"Font file not found at path: {font_path}")
        raise FileNotFoundError(f"Font file not found at path: {font_path}")
    pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))

    # Create a SimpleDocTemplate
    doc = SimpleDocTemplate(buffer,
                            pagesize=letter,
                            rightMargin=72,
                            leftMargin=72,
                            topMargin=72,
                            bottomMargin=18)

    # Define styles using DejaVu Sans
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(name='DejaVu',
                       fontName='DejaVuSans',
                       fontSize=12,
                       leading=15))

    flowables = []

    # Split the text into paragraphs based on double newline
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        # Create a Paragraph object and add to flowables
        p = Paragraph(para, styles['DejaVu'])
        flowables.append(p)
        flowables.append(Spacer(1, 0.2 * inch))  # Add space between paragraphs

    # Build the PDF
    try:
        doc.build(flowables)
        logging.info("PDF generated successfully.")
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")
        raise e

    buffer.seek(0)
    return buffer


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    try:
        youtube_url = request.form['youtube_url']
        repair = request.form.get('repair') == 'true'
        video_id = extract_video_id(youtube_url)
        if not video_id:
            error = "Invalid YouTube URL."
            logging.error(error)
            return jsonify({'status': 'error', 'message': error})

        transcript = get_transcript(video_id)
        if not transcript:
            error = "Could not retrieve transcript for this video."
            logging.error(error)
            return jsonify({'status': 'error', 'message': error})

        # Initialize the LLM with tracing enabled
        with tracing_v2_enabled():
            llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5)

            if repair:
                logging.info("Repairing transcript...")
                repaired_transcript = repair_transcript(transcript, llm)
            else:
                repaired_transcript = transcript

            logging.info("Translating transcript...")
            translated_text = translate_to_turkish(repaired_transcript, llm)

        logging.info("Processing completed successfully.")
        # Store the translated text in the session instead of passing it as a URL parameter
        session['translated_text'] = translated_text
        return jsonify({
            'status': 'success',
            'message': 'Translation completed'
        })

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        return jsonify({
            'status':
            'error',
            'message':
            'An error occurred during processing. Please try again.'
        })


@app.route('/result', methods=['GET', 'POST'])
def result():
    # Retrieve the translated text from the session
    translated_text = session.get('translated_text')
    if not translated_text:
        return redirect(url_for('index'))
    # Clear the session data after retrieving it
    session.pop('translated_text', None)
    return render_template('result.html', translated_text=translated_text)


@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    translated_text = request.form['translated_text']
    try:
        pdf_buffer = generate_pdf(translated_text)
    except Exception as e:
        logging.error(f"Failed to generate PDF: {e}")
        return redirect(url_for('index'))
    return send_file(pdf_buffer,
                     as_attachment=True,
                     download_name='translated_transcript.pdf',
                     mimetype='application/pdf')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

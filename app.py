# === Standard Python & Streamlit Imports ===
import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import base64

# === CrewAI Core Classes ===
from crewai import Agent, Task, Crew

# === LangChain-Compatible Tools ===
from crewai_tools import CodeInterpreterTool  # CrewAI-native code execution tool
from crewai_tools import SerperDevTool  # Web search tool
from crewai.tools import BaseTool  # Custom summarization tool
from pydantic import BaseModel, Field  # For defining tool attributes

import openai  # For summarization via GPT

# === Load Environment Variables from .env ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# === Error handling for missing keys ===
if not openai.api_key:
    st.error(
        "Chave de API do OpenAI ausente! Certifique-se de que OPENAI_API_KEY esteja definida no arquivo .env ou nas variáveis de ambiente."
    )
    st.info(
        "Você pode obter uma chave de API em https://platform.openai.com/api-keys"
    )
    st.stop()

if not serper_api_key:
    st.error(
        "Chave de API do Serper ausente! Certifique-se de que SERPER_API_KEY esteja definida no arquivo .env ou nas variáveis de ambiente."
    )
    st.info("Você pode obter uma chave de API em https://serper.dev/")
    st.stop()


# === Função para criar PDF ===
def create_pdf(content):
    # Converter o conteúdo para string, caso seja um objeto
    if not isinstance(content, str):
        content = str(content)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    flowables = []

    # Título
    title_style = styles["Title"]
    flowables.append(
        Paragraph("Resultado do Fluxo de Trabalho de Agentes IA", title_style))
    flowables.append(Spacer(1, 12))

    # Conteúdo principal
    normal_style = styles["Normal"]
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            # Substituir caracteres que podem causar problemas no PDF
            para = para.replace("&",
                                "&amp;").replace("<",
                                                 "&lt;").replace(">", "&gt;")
            try:
                flowables.append(Paragraph(para, normal_style))
                flowables.append(Spacer(1, 6))
            except Exception:
                # Se houver erro ao adicionar o parágrafo, tente adicionar como texto simples
                flowables.append(
                    Paragraph(f"[Conteúdo não renderizável]", normal_style))

    # Construir o documento
    doc.build(flowables)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# === Função de download ===
def get_pdf_download_link(pdf_bytes, filename="resultado_agentes_ia.pdf"):
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Baixar PDF</a>'
    return href


# === Initialize Tools ===

# Web Search Tool using Serper (Google-like search)
search_tool = SerperDevTool(api_key=serper_api_key)

# Code Execution Tool using built-in Python interpreter (safe + multi-line)
code_tool = CodeInterpreterTool()


# Match the actual key CrewAI is passing ("description")
class SummarizeToolInput(BaseModel):
    description: str = Field(..., description="Text to summarize")


class SummarizeTool(BaseTool):
    name: str = "Summarizer"
    description: str = "Resume textos usando OpenAI GPT"
    args_schema = SummarizeToolInput  # ✅ matches the field name

    def _run(self, description: str) -> str:
        """Resume o texto usando GPT."""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"Resumir este texto em português:\n\n{description}"
                }],
                temperature=0.3,
                max_tokens=200)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Erro de resumo: {e}"


# Initialize summarizer
summarize_tool = SummarizeTool()

# === Tool Mapping ===
TOOL_MAP = {
    "search": search_tool,
    "code": code_tool,
    "summarize": summarize_tool
}

# === Streamlit App Setup ===
st.set_page_config(page_title="🧠 Construtor de Fluxo de Agentes IA",
                   layout="centered")

# === Theme Management ===
# Define theme palettes
THEME_PALETTES = {
    "Default Azul": {
        "dark": {
            "primary": "#0066CC",
            "background": "#000000",
            "secondary_background": "#111111",
            "text": "#FFFFFF",
            "highlight_text": "#FFFFFF",
            "secondary_text": "#DDD",
            "border": "#333333"
        },
        "light": {
            "primary": "#0066CC",
            "background": "#FFFFFF",
            "secondary_background": "#F0F2F6",
            "text": "#333333",
            "highlight_text": "#000000",
            "secondary_text": "#666666",
            "border": "#DDDDDD"
        }
    },
    "Roxo": {
        "dark": {
            "primary": "#8A2BE2",
            "background": "#0A0A0A",
            "secondary_background": "#1A1A1A",
            "text": "#FFFFFF",
            "highlight_text": "#FFFFFF",
            "secondary_text": "#DDD",
            "border": "#333333"
        },
        "light": {
            "primary": "#8A2BE2",
            "background": "#FFFFFF",
            "secondary_background": "#F5F0FF",
            "text": "#333333",
            "highlight_text": "#000000",
            "secondary_text": "#666666",
            "border": "#E0D1FF"
        }
    },
    "Verde": {
        "dark": {
            "primary": "#00A86B",
            "background": "#0A0A0A",
            "secondary_background": "#1A1A1A",
            "text": "#FFFFFF",
            "highlight_text": "#FFFFFF",
            "secondary_text": "#DDD",
            "border": "#333333"
        },
        "light": {
            "primary": "#00A86B",
            "background": "#FFFFFF",
            "secondary_background": "#F0FFF5",
            "text": "#333333",
            "highlight_text": "#000000",
            "secondary_text": "#666666",
            "border": "#CCFFE0"
        }
    }
}

# Initialize theme state if not already present
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "dark"
if 'theme_palette' not in st.session_state:
    st.session_state.theme_palette = "Default Azul"

# Get current theme colors based on selected palette and mode
def get_current_theme():
    return THEME_PALETTES[st.session_state.theme_palette][st.session_state.theme_mode]

# Function to toggle theme mode
def toggle_theme_mode():
    st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"

# Function to change theme palette
def change_theme_palette(new_palette):
    # Altera a paleta de cores baseado na seleção
    st.session_state.theme_palette = new_palette

# Obter o tema atual e gerar CSS dinâmico
current_theme = get_current_theme()

# Estilo CSS personalizado baseado no tema atual
theme_css = f"""
<style>
    body {{
        color: {current_theme["text"]};
        background-color: {current_theme["background"]};
    }}
    
    .container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }}
    
    .logo-img {{
        max-width: 220px;
        height: auto;
        margin-right: 20px;
    }}
    
    .title-container {{
        flex-grow: 1;
    }}
    
    .main-title {{
        font-size: 32px;
        font-weight: bold;
        color: {current_theme["highlight_text"]};
    }}
    
    .subtitle {{
        font-size: 16px;
        color: {current_theme["secondary_text"]};
        margin-top: 10px;
    }}
    
    .highlight {{
        color: {current_theme["primary"]};
        font-weight: bold;
    }}
    
    .stButton button {{
        background-color: {current_theme["primary"]};
        color: white;
        font-weight: bold;
    }}
    
    /* Tema específico para tabela e outros elementos */
    .stDataFrame, .stTable {{
        background-color: {current_theme["secondary_background"]};
        color: {current_theme["text"]};
    }}
    
    /* Estilos para links */
    a {{
        color: {current_theme["primary"]};
        text-decoration: none;
    }}
    
    a:hover {{
        text-decoration: underline;
    }}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

# Header com logo e título
st.markdown("""
<div class="container">
    <img src="static/images/logo.png" class="logo-img">
    <div class="title-container">
        <div class="main-title">🧠 Construtor de Fluxo de Agentes IA</div>
        <div class="subtitle">
            Construa e execute uma equipe personalizada de agentes de IA usando 
            <span class="highlight">CrewAI</span>.
            Cada agente pode ter funções, objetivos e ferramentas diferentes.
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True)

# Estilização da sidebar com cores do tema atual
sidebar_css = f"""
<style>
    .sidebar-header {{
        font-size: 24px;
        font-weight: bold;
        color: {current_theme["highlight_text"]};
        margin-bottom: 20px;
    }}
    
    .tool-item {{
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 5px;
        background-color: {current_theme["secondary_background"]};
    }}
    
    .tool-name {{
        color: {current_theme["primary"]};
        font-weight: bold;
        font-size: 18px;
    }}
    
    .tool-description {{
        color: {current_theme["secondary_text"]};
        font-size: 14px;
        margin-top: 5px;
    }}
    
    .sidebar-footer {{
        position: fixed;
        bottom: 20px;
        left: 20px;
        font-size: 12px;
        color: {current_theme["secondary_text"]};
    }}
    
    /* Estilização específica para o tema claro/escuro */
    .theme-toggle-container {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-bottom: 20px;
        padding: 10px;
        border-radius: 5px;
        background-color: {current_theme["secondary_background"]};
    }}
</style>
"""

st.sidebar.markdown(sidebar_css, unsafe_allow_html=True)

# Obter o tema atual
current_theme = get_current_theme()

# Componente para seleção de tema na sidebar
st.sidebar.markdown(
    '<div class="sidebar-header">🎨 Personalização</div>',
    unsafe_allow_html=True)

# Containers para os controles de tema
theme_mode_col, theme_palette_col = st.sidebar.columns(2)

# Botão para alternar entre modo claro e escuro
with theme_mode_col:
    mode_icon = "☀️" if st.session_state.theme_mode == "dark" else "🌙"
    mode_text = "Modo Claro" if st.session_state.theme_mode == "dark" else "Modo Escuro"
    if st.button(f"{mode_icon} {mode_text}", key="theme_toggle"):
        toggle_theme_mode()
        st.rerun()

# Seletor de paleta de cores
with theme_palette_col:
    selected_palette = st.selectbox(
        "Cores",
        options=list(THEME_PALETTES.keys()),
        index=list(THEME_PALETTES.keys()).index(st.session_state.theme_palette),
        key="palette_selector"
    )
    if selected_palette != st.session_state.theme_palette:
        change_theme_palette(selected_palette)
        st.rerun()

st.sidebar.markdown("---")

# Add description for each tool
st.sidebar.markdown(
    '<div class="sidebar-header">🧰 Ferramentas Disponíveis</div>',
    unsafe_allow_html=True)

# Ferramentas com descrição melhorada
st.sidebar.markdown("""
<div class="tool-item">
    <div class="tool-name">🔍 search</div>
    <div class="tool-description">Ferramenta de pesquisa web usando Serper (similar ao Google)</div>
</div>

<div class="tool-item">
    <div class="tool-name">💻 code</div>
    <div class="tool-description">Execute código Python para resolver problemas complexos</div>
</div>

<div class="tool-item">
    <div class="tool-name">📝 summarize</div>
    <div class="tool-description">Resumir texto usando OpenAI GPT</div>
</div>
""",
                    unsafe_allow_html=True)

# Footer na sidebar
st.sidebar.markdown("""
<div class="sidebar-footer">
    © 2024 StartSe Consulting - Powered by CrewAI
</div>
""",
                    unsafe_allow_html=True)

# === Task Input ===
task_description = st.text_input(
    "📝 Em que os agentes devem trabalhar?",
    value="Pesquise os últimos avanços em IA generativa e os resuma.")

# === Custom Styles for Input Elements ===
input_css = f"""
<style>
    /* Styling for all inputs */
    .stTextInput > div > div > input {{
        background-color: {current_theme["secondary_background"]};
        color: {current_theme["text"]};
        border: 1px solid {current_theme["border"]};
    }}
    
    .stTextArea > div > div > textarea {{
        background-color: {current_theme["secondary_background"]};
        color: {current_theme["text"]};
        border: 1px solid {current_theme["border"]};
    }}
    
    /* Styling for multi-select */
    .stMultiSelect {{
        background-color: {current_theme["secondary_background"]};
    }}
    
    /* Styling for expander */
    .streamlit-expanderHeader {{
        background-color: {current_theme["primary"]} !important;
        color: white !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }}
    
    /* Custom launch button styling */
    .launch-btn {{
        background-color: {current_theme["primary"]};
        color: white;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 20px;
        margin-bottom: 20px;
        width: 100%;
        text-align: center;
    }}
    
    .launch-btn:hover {{
        background-color: {current_theme["primary"] + '99'};
    }}
    
    /* Divider styling */
    hr {{
        border-color: {current_theme["border"]};
    }}
    
    /* Configuration section header */
    .config-header {{
        color: {current_theme["highlight_text"]};
        background-color: {current_theme["secondary_background"]};
        padding: 10px 15px;
        border-radius: 5px;
        border-left: 3px solid {current_theme["primary"]};
        margin-top: 20px;
        margin-bottom: 20px;
        font-size: 20px;
    }}
    
    /* Ajustes para elementos específicos do streamlit */
    .stSlider {{
        margin-bottom: 20px;
    }}
    
    /* Seletores */
    .stSelectbox > div > div {{
        background-color: {current_theme["secondary_background"]};
        color: {current_theme["text"]};
    }}
</style>
"""

st.markdown(input_css, unsafe_allow_html=True)

# === Number of Agents ===
num_agents = st.slider("👥 Número de agentes",
                       min_value=2,
                       max_value=5,
                       value=3)

# === Agent Configuration UI ===
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="config-header">⚙️ Configure Cada Agente</div>',
            unsafe_allow_html=True)

agent_configs = []

# For each agent, gather role, goal, and tools via UI
for i in range(num_agents):
    with st.expander(f"Agente {i+1}"):
        role = st.text_input(f"🔧 Função do Agente {i+1}",
                             value=f"Agente {i+1}")
        goal = st.text_area(f"🎯 Objetivo do Agente {i+1}",
                            value=f"Auxiliar com: {task_description}")
        tools = st.multiselect(f"🧰 Ferramentas do Agente {i+1}",
                               options=list(TOOL_MAP.keys()),
                               default=["search"])
        agent_configs.append({"role": role, "goal": goal, "tools": tools})

# === Launch Button ===
st.markdown("<div style='text-align: center; margin: 30px 0;'>",
            unsafe_allow_html=True)
start_button = st.button("🚀 Iniciar Agentes Sequencialmente",
                         key="launch_btn",
                         help="Iniciar processamento dos agentes",
                         use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Use the button state for logic control
if start_button:
    st.info("🛠️ Iniciando seus agentes de IA sequencialmente...")

    agents = []

    # Step 1: Build Agent objects
    for config in agent_configs:
        selected_tools = [
            TOOL_MAP[t] for t in config["tools"] if t in TOOL_MAP
        ]

        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            tools=selected_tools,
            backstory=f"{config['role']} está colaborando neste projeto.",
            verbose=True)
        agents.append(agent)

    # Step 2: Sequential execution
    current_input = task_description  # Initial input to first task
    results = []  # Store intermediate results

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, agent in enumerate(agents):
        task = Task(description=f"{current_input}",
                    agent=agent,
                    expected_output=f"Resultado do {agent.role}")

        crew = Crew(agents=[agent], tasks=[task], verbose=True)

        status_text.text(f"🤖 Agente {i+1} ({agent.role}) está trabalhando...")

        try:
            output = crew.kickoff()
            results.append((agent.role, output))
            current_input = output  # Pass to next agent
        except Exception as e:
            error_message = f"Erro com {agent.role}: {str(e)}"
            st.error(error_message)
            results.append((agent.role, error_message))
            # Continue with next agent using last successful output

        # Update progress
        progress_percentage = (i + 1) / len(agents)
        progress_bar.progress(progress_percentage)

    # Step 3: Display final result
    status_text.text("✅ Todos os agentes completaram suas tarefas!")

    # Estilo para a área de resultados baseado no tema atual
    results_css = f"""
    <style>
        .results-header {{
            background-color: {current_theme["primary"]};
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            margin-top: 30px;
            margin-bottom: 20px;
            font-size: 20px;
            font-weight: bold;
        }}
        
        .results-container {{
            background-color: {current_theme["secondary_background"]};
            padding: 20px;
            border-radius: 5px;
            border-left: 5px solid {current_theme["primary"]};
        }}
        
        .download-btn {{
            display: inline-block;
            background-color: {current_theme["primary"]};
            color: white;
            padding: 8px 15px;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }}
        
        .download-btn:hover {{
            background-color: {current_theme["primary"] + '99'};
        }}
        
        .agent-result {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: {current_theme["secondary_background"]};
            border-radius: 5px;
            border-left: 3px solid {current_theme["primary"]};
        }}
        
        .agent-name {{
            font-size: 18px;
            font-weight: bold;
            color: {current_theme["primary"]};
            margin-bottom: 10px;
        }}
    </style>
    """
    
    st.markdown(results_css, unsafe_allow_html=True)

    st.markdown(
        '<div class="results-header">✅ Todos os agentes completaram suas tarefas!</div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="results-header">📄 Resultado Final (do último agente)</div>',
        unsafe_allow_html=True)
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.write(current_input)
    st.markdown('</div>', unsafe_allow_html=True)

    # Generate PDF with the result
    pdf_bytes = create_pdf(current_input)
    st.markdown(
        f'<a href="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" download="resultado_agentes_ia.pdf" class="download-btn">📥 Baixar Resultado em PDF</a>',
        unsafe_allow_html=True)

    # Optional: Show step-by-step outputs
    with st.expander("🧾 Resultados Completos dos Agentes"):
        all_outputs = ""
        for role, output in results:
            st.markdown(
                f'<div class="agent-result"><div class="agent-name">{role}</div>',
                unsafe_allow_html=True)
            st.write(output)
            st.markdown('</div>', unsafe_allow_html=True)

            # Converter o output para string, caso seja um objeto
            output_str = str(output) if not isinstance(output, str) else output
            all_outputs += f"## {role}\n\n{output_str}\n\n"

        # Option to download all outputs
        complete_pdf = create_pdf(all_outputs)
        st.markdown(
            f'<a href="data:application/pdf;base64,{base64.b64encode(complete_pdf).decode()}" download="resultados_completos_agentes_ia.pdf" class="download-btn">📥 Baixar Todos os Resultados em PDF</a>',
            unsafe_allow_html=True)

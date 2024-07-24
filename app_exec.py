import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import boto3
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
import openai
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Set up NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

# Check if the data exists before downloading
if not os.path.exists(os.path.join(nltk_data_dir, 'corpora', 'wordnet')):
    nltk.download('wordnet', download_dir=nltk_data_dir)

# Set up Boto3 session
boto3.setup_default_session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('REGION_NAME')
)

# Set page configuration
st.set_page_config(page_title="TargetList Recommender", page_icon=":dart:")

# OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to load dataset from S3
@st.cache_data
def load_data():
    s3 = boto3.client('s3')
    bucket_name = 'iptpdatabase'
    file_key = 'your_dataset2.csv'
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    df.fillna('', inplace=True)
    df['combined_industries'] = df['Industry'] + ' ' + df['Affinity_Industries']
    df['Client_industry'] = df['Client_industry'].replace('MarTech', 'MarTech (Marketing Technology)')
    df['Client_combined'] = df.apply(lambda row: row['Client_industry'] if pd.isna(row['Client_Subindustry']) else f"{row['Client_industry']} ({row['Client_Subindustry']})", axis=1)
    return df

# Function to get relevant industries using OpenAI GPT
def get_relevant_industries(description):
    prompt = f"""
    Identify and rank the most relevant and specific industries and keywords for the following company description, excluding broad categories like 'Technology' and 'SaaS'. Give me only a list of the industries identified without repeating the word industry, sorted from most relevant and specific:
    {description}

    Relevant Industries:
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    relevant_part = response['choices'][0]['message']['content'].strip()
    industries = [line.strip() for line in relevant_part.split("\n") if line.strip()]
    return industries

# Function to load the Sentence Transformer model
@st.cache_resource
def load_model(model_name):
    model = SentenceTransformer(model_name)
    return model

model_name = "all-MiniLM-L6-v2"
model = load_model(model_name)

# Function to get embeddings
def get_embeddings(text, model):
    embeddings = model.encode([text])
    return embeddings

# Advanced algorithm for smarter recommendations
def advanced_alg(df, client_industries, target_industries, excluded_dossiers, industry_weight, affinity_weight, tier_weight, similarity_weight):
    industry_embeddings = model.encode(df['combined_industries'].tolist())
    
    client_industry_embeddings = [get_embeddings(ind, model)[0] for ind in client_industries]
    target_industry_embeddings = [get_embeddings(ind, model)[0] for ind in target_industries]

    client_sim_scores = cosine_similarity(industry_embeddings, client_industry_embeddings).max(axis=1)
    target_sim_scores = cosine_similarity(industry_embeddings, target_industry_embeddings).max(axis=1)
    
    df['similarity_score'] = (client_sim_scores * industry_weight) + (target_sim_scores * affinity_weight)
    df['inverted_tier'] = 5 - df['Tiering']
    df['combined_score'] = df['similarity_score'] * similarity_weight + df['inverted_tier'] * tier_weight

    filtered_df = df[df['combined_score'] > 0]
    filtered_df = filtered_df[~filtered_df['Dossier'].isin(excluded_dossiers)]
    scored_targets = filtered_df.sort_values(by='combined_score', ascending=False).drop_duplicates('Target').reset_index(drop=True)
    
    return scored_targets[['Target','Industry','Affinity_Industries', 'Dossier', 'Client_combined' , 'Tiering', 'similarity_score', 'combined_score']]

# Function to add GPT scores and comments to the dataset
def add_gpt_scores_and_comments(recommendations, description):
    def evaluate_relevance(target):
        prompt = f"""
        Based on the following company description: "{description}", evaluate the relevance of this potential M&A target based in database industries and your knowledge of the target:
        Target: {target['Target']}
        Industry: {target['Industry IPTP']}
        Affinity Industries: {target['Affinity_Industries']}
        Client Combined: {target['Client_combined']}

        Provide a score between 0 to 10.
        
        Format your response as:
        Score: [0-10]
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        result = response['choices'][0]['message']['content'].strip().split('\n')
        score = None
        comment = None
        for line in result:
            if 'score' in line.lower():
                score = line.split(":")[-1].strip()
            if 'comment' in line.lower():
                comment = line.split(":")[-1].strip()
        return score, comment

    recommendations[['scoreGPT', 'Comments']] = recommendations.apply(evaluate_relevance, axis=1, result_type='expand')
    recommendations['scoreGPT'] = pd.to_numeric(recommendations['scoreGPT'], errors='coerce')
    recommendations = recommendations.dropna(subset=['scoreGPT'])
    recommendations = recommendations.sort_values(by='scoreGPT', ascending=False).reset_index(drop=True)
    return recommendations

# Display logo, title, and icon side by side
col1, col2, col3 = st.columns([2, 10, 2])
with col1:
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)  # Add margin
    st.image('images/IPTP.png', width=90)
with col2:
    st.markdown("<h1 style='margin: 0; text-align: center;'>Target List Recommendation</h1>", unsafe_allow_html=True)
with col3:
    st.markdown("<h2 style='margin: 0; text-align: right; font-size: 80px;'>🎯</h2>", unsafe_allow_html=True)

# Load data
df = load_data()

# Sidebar for parameter selection
st.sidebar.title("Parameters")
tier_weight = st.sidebar.slider('Tier Weight:', 0.0, 1.0, 0.3)
similarity_weight = st.sidebar.slider('Similarity Weight:', 0.0, 1.0, 0.7)
industry_weight = st.sidebar.slider('Industry Weight:', 0.0, 1.0, 0.5)
affinity_weight = st.sidebar.slider('Affinity Industry Weight:', 0.0, 1.0, 0.5)
excluded_dossiers = st.sidebar.multiselect('Exclude Dossiers:', df['Dossier'].unique())
num_targets = st.sidebar.number_input('Number of Targets to Generate:', min_value=1, max_value=1000, value=200)

# User inputs
option = st.sidebar.radio("Select Input Method", ('Description', 'Industries'))

input_description = ""
input_industries = ""
recommendations = pd.DataFrame()

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()
if 'relevant_industries' not in st.session_state:
    st.session_state.relevant_industries = []
if 'selected_industries' not in st.session_state:
    st.session_state.selected_industries = []
if 'input_description' not in st.session_state:
    st.session_state.input_description = ""
if 'input_industries' not in st.session_state:
    st.session_state.input_industries = ""
if 'recommendations_with_gpt' not in st.session_state:
    st.session_state.recommendations_with_gpt = pd.DataFrame()
if 'new_industry' not in st.session_state:
    st.session_state.new_industry = ""

if option == 'Description':
    st.session_state.input_description = st.text_area('Enter the Company Description:', st.session_state.input_description)
    if st.button('Generate industries and dataset'):
        if st.session_state.input_description:
            st.write('Analyzing the Company Description:')
            st.session_state.relevant_industries = get_relevant_industries(st.session_state.input_description)
            st.session_state.selected_industries = st.session_state.relevant_industries
else:
    st.session_state.input_industries = st.text_area('Enter the Relevant Industries (separated by commas):', st.session_state.input_industries)
    st.session_state.selected_industries = [ind.strip() for ind in st.session_state.input_industries.split(',')]

# Always show the identified industries and allow modification
if st.session_state.relevant_industries:
    st.info(f"**Identified Relevant Industries:** {', '.join(st.session_state.relevant_industries)}")
    st.session_state.selected_industries = st.multiselect('Modify Relevant Industries:', st.session_state.selected_industries, default=st.session_state.selected_industries)

# Function to add new industry
def add_new_industry():
    if st.session_state.new_industry:
        if st.session_state.new_industry not in st.session_state.selected_industries:
            st.session_state.selected_industries.append(st.session_state.new_industry)
            st.session_state.new_industry = ""

# Text input and button for adding a new industry
st.session_state.new_industry = st.text_input("Add a new industry:", st.session_state.new_industry)
st.button("Add Industry", on_click=add_new_industry)

if st.session_state.selected_industries:
    st.session_state.recommendations = advanced_alg(df, st.session_state.selected_industries, st.session_state.selected_industries, excluded_dossiers, industry_weight, affinity_weight, tier_weight, similarity_weight)[:num_targets]
    
    st.session_state.recommendations = st.session_state.recommendations.rename(columns={'Industry': 'Industry IPTP'})
    st.write(f"<h3>Generated Dataset ({len(st.session_state.recommendations)})</h3>", unsafe_allow_html=True)
    st.dataframe(st.session_state.recommendations)

    if st.button('Add GPT-4 Scores and Comments'):
        with st.spinner('Evaluating relevance using GPT-4...'):
            st.session_state.recommendations_with_gpt = add_gpt_scores_and_comments(st.session_state.recommendations, st.session_state.input_description)
        st.write(f"<h3>Dataset with GPT-4 Scores ({len(st.session_state.recommendations_with_gpt)})</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.recommendations_with_gpt)

        if len(st.session_state.recommendations_with_gpt) == 0:
            st.write("No relevant targets found after further filtering.")

        # Allow user to download the recommendations as a CSV file
        csv = st.session_state.recommendations_with_gpt.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='recommendations.csv',
            mime='text/csv',
            key='download-recommendations'
        )

if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)

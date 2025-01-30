import streamlit as st
import pandas as pd
import openai  # Ensure you have the openai package installed
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
import folium
from playwright.sync_api import sync_playwright
from googlesearch import search
import time
from geopy.geocoders import GoogleV3
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Environment variables and API keys
MAPS_KEY = "YOUR_MAPS_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
SEARCH_API_KEY = "YOUR_SEARCH_KEY"
CSE_ID = "YOUR_CSE_ID"
CSE_ID_LINKEDIN = "YOUR_LINKEDIN_CSE_ID"

# Initialise OpenAI
openai.api_key = OPENAI_API_KEY


# Function Definitions
def go_forward():
    if st.session_state["step"] < 6:
        st.session_state["step"] += 1


def go_back():
    if st.session_state["step"] > 1:
        st.session_state["step"] -= 1


def extract_keywords_with_gpt(service_description):
    """
    Use GPT to extract keywords from the service description.
    """
    try:
        prompt = (
            f"Extract 3-5 concise keywords from the following service description. "
            f"Ensure the keywords are relevant to the service and avoid overly generic terms:\n\n"
            f"Service Description: {service_description}\n\n"
            f"Keywords:"
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a highly accurate keyword extractor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.2
        )

        keywords = response.choices[0].message["content"].strip()
        return keywords.split(", ")

    except Exception as e:
        st.error(f"Error extracting keywords with GPT: {str(e)}")
        return []


def find_businesses_with_websites(location, audience, radius):
    """
    Find businesses with websites using Google Maps API.
    """
    try:
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query": f"{audience} near {location}",
            "radius": radius,
            "key": MAPS_KEY,
        }

        response = requests.get(url, params=params)
        results = response.json().get("results", [])

        businesses = []
        for result in results[:10]:
            place_id = result.get("place_id")
            details_url = f"https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {
                "place_id": place_id,
                "fields": "name,formatted_address,website",
                "key": MAPS_KEY,
            }

            details_response = requests.get(details_url, params=details_params).json()
            details = details_response.get("result", {})

            if details.get("website"):
                businesses.append({
                    "name": details.get("name"),
                    "address": details.get("formatted_address"),
                    "website": details.get("website"),
                })

        return businesses
    except Exception as e:
        return f"Error finding businesses: {str(e)}"


def scrape_contacts_from_websites(websites):
    """
    Scrape emails and contacts from websites using BeautifulSoup.
    """
    contacts = {}
    email_pattern = re.compile(r"[^@]+@[^@]+\.[^@]+")

    for website in websites:
        try:
            response = requests.get(website, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            email_list = []

            for mail in soup.find_all("a", href=True):
                href = mail.get("href")
                if href.startswith("mailto:"):
                    email = href.replace("mailto:", "").strip()
                    if email_pattern.match(email):
                        email_list.append(email)

            if email_list:
                contacts[website] = list(set(email_list))
            else:
                contacts[website] = ["No email found"]
        except Exception as e:
            contacts[website] = [f"Error: {str(e)}"]

    return contacts


def generate_email_with_gpt(template_type, username, service, audience, contact_name, context, description=None):
    """
    Generate an email using GPT-3.5 based on provided parameters.
    """
    try:
        prompt = (
            f"You are an expert email copywriter. Write a {template_type.lower()} email "
            f"from {username} who specialises in {service}. "
            f"Their target audience is {audience}, and the email is addressed to {contact_name}. "
            f"Context: {context} "
        )

        if description and len(description.strip()) > 0:
            prompt += f"Additional context: {description}. "

        prompt += (
            f"The email should:\n"
            f"1. Include a clear subject line (5-7 words)\n"
            f"2. Be no more than 80 words\n"
            f"3. Have a friendly yet professional tone\n"
            f"4. Include a clear call to action\n"
            f"5. Be written in British English\n"
            f"6. Be structured in 2-3 short paragraphs\n"
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional email copywriter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        raise Exception(f"Error in email generation: {str(e)}")


def validate_urls(urls):
    """
    Validate a list of URLs by checking their response status.
    """
    valid_urls = []
    for url in urls:
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                valid_urls.append(url)
        except Exception:
            continue
    return valid_urls


def expand_keywords(input_keyword):
    """
    Expand a keyword into related terms using GPT.
    """
    try:
        prompt = (
            f"Generate 5 related keywords or synonyms for: '{input_keyword}'. "
            f"Make them simple and relevant for searches."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a keyword generation expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        keywords = response.choices[0].message["content"].strip()
        return [kw.strip() for kw in keywords.split(",") if kw.strip()]
    except Exception as e:
        st.error(f"Error expanding keywords: {str(e)}")
        return [input_keyword]


def get_top_companies(industry, max_companies=2):
    """Find top companies in a given industry using Google Custom Search"""
    try:
        query = f"top companies in {industry} industry UK"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": SEARCH_API_KEY,
            "cx": CSE_ID,
            "num": max_companies
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        companies = []
        if "items" in data:
            for item in data["items"]:
                # Extract company name from title
                company_name = item.get("title", "").split("-")[0].strip()
                companies.append(company_name)

        return companies[:max_companies]
    except Exception as e:
        st.error(f"Error finding top companies: {str(e)}")
        return []


def fetch_and_summarise(urls):
    """
    Fetch content from URLs and summarise it using GPT.
    """
    summaries = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            post_content = " ".join([p.text for p in soup.find_all("p")]).strip()[:1000]

            if post_content:
                prompt = f"Summarise this post: {post_content}"
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50
                )
                summaries.append({"url": url, "summary": response.choices[0].message["content"].strip()})
            else:
                summaries.append({"url": url, "summary": "No content found."})
        except Exception as e:
            summaries.append({"url": url, "summary": f"Error: {str(e)}"})
    return summaries


def google_custom_search(query, api_key, cse_id, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error if the request fails
        data = response.json()
        items = data.get("items", [])
        return [{"title": item.get("title"), "link": item.get("link")} for item in items]
    except requests.exceptions.RequestException as e:
        st.error(f"Search Error: {e}")
        return []


def google_search_mumsnet(query, num_results=5):
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        # Make the search query more flexible
        search_query = f'site:mumsnet.com {query}'  # Removed strict quotes

        params = {
            "q": search_query,
            "key": os.getenv("GOOGLE_SEARCH_API_KEY"),
            "cx": os.getenv("GOOGLE_CSE_ID_MUMSNET"),
            "num": num_results
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])

        return [{"title": item.get("title"), "link": item.get("link")} for item in items]
    except Exception as e:
        st.error(f"Error during Mumsnet search: {str(e)}")
        return []


def google_search_facebook(role, num_results=5):
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        expanded_keywords = expand_keywords(role)
        all_results = []

        # More natural, conversational search phrases
        base_phrases = [
            "looking for recommendations",
            "need help with",
            "can anyone suggest",
            "searching for",
            "anyone know",
            "recommendations for"
        ]

        for kw in expanded_keywords:
            for phrase in base_phrases:
                search_query = f'site:facebook.com {phrase} {kw}'  # Removed strict quotes for broader matches
                params = {
                    "q": search_query,
                    "key": os.getenv("GOOGLE_SEARCH_API_KEY"),
                    "cx": os.getenv("GOOGLE_CSE_ID_FACEBOOK"),
                    "num": num_results
                }

                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                items = data.get("items", [])

                if items:
                    all_results.extend([{"title": item.get("title"), "link": item.get("link")} for item in items])

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for result in all_results:
            if result["link"] not in seen:
                seen.add(result["link"])
                unique_results.append(result)

        return unique_results
    except Exception as e:
        st.error(f"Error during Facebook search: {str(e)}")
        return []


def generate_related_search_terms(service):
    """
    Use GPT to generate broader, more conversational search terms for a service
    """
    try:
        prompt = (
            f"Generate 8 different ways people might discuss or search for '{service}' on social media. Include:\n"
            f"1. Casual, conversational phrases (e.g., 'anyone know a good...')\n"
            f"2. Problem-based phrases (e.g., 'struggling with...')\n"
            f"3. Recommendation requests (e.g., 'can anyone recommend...')\n"
            f"4. Price inquiries (e.g., 'how much should I pay for...')\n"
            f"5. Quality-focused queries (e.g., 'best {service} in...')\n"
            f"Make the phrases natural and conversational, as if someone was posting on social media.\n"
            f"Format: Return only the search phrases, one per line."
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in social media conversations and search behavior."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Increased for more variety
            max_tokens=200
        )

        search_terms = response.choices[0].message["content"].strip().split('\n')
        return [term.strip() for term in search_terms if term.strip()]
    except Exception as e:
        st.error(f"Error generating search terms: {str(e)}")
        return []


def process_user_input(input_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that identifies roles, related job titles, and industries from user input."},
            {"role": "user", "content": f"""
            Analyze this request: '{input_text}'.
            1. Identify the primary role(s) mentioned.
            2. Suggest 1-3 related job titles.
            3. Specify the industry associated with these roles.
            Format the response as follows:
            Roles:
            - Role 1
            Related Titles:
            - Title 1
            - Title 2
            Industry:
            - [Industry Name]"""},
        ]
    )
    return response["choices"][0]["message"]["content"]


def extract_roles_and_industry(gpt_response):
    roles_match = re.search(r"Roles:(.*?)(Related Titles|Industry):", gpt_response, re.DOTALL)
    roles = [line.strip("- ").strip() for line in roles_match.group(1).split("\n") if
             line.strip()] if roles_match else []

    titles_match = re.search(r"Related Titles:(.*?)(Industry):", gpt_response, re.DOTALL)
    related_titles = [line.strip("- ").strip() for line in titles_match.group(1).split("\n") if
                      line.strip()] if titles_match else []

    industry_match = re.search(r"Industry:(.*)", gpt_response, re.DOTALL)
    industry = industry_match.group(1).strip() if industry_match else "General"

    return roles, related_titles, industry


def linkedin_search_pse(query, company_name, max_results=3):
    """Search for LinkedIn profiles using PSE"""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": SEARCH_API_KEY,
        "cx": CSE_ID_LINKEDIN,  # Using LinkedIn specific CSE
        "num": max_results
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json()
            profiles = []

            if "items" in results:
                for item in results["items"]:
                    if "linkedin.com/in/" in item.get("link", ""):
                        profile = {
                            "Name": item.get("title", "").split("-")[0].strip(),
                            "Title": item.get("title", "").split("-")[1].strip() if "-" in item.get("title",
                                                                                                    "") else "",
                            "Company": company_name,
                            "Source": item.get("link", "")
                        }
                        profiles.append(profile)
            return profiles
        else:
            st.write(f"LinkedIn search error: {response.status_code}")
            return []
    except Exception as e:
        st.write(f"Error during LinkedIn search: {str(e)}")
        return []


def company_website_search(company_name):
    """Search for company website using Google Search API"""
    # Get API key from environment
    api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
    search_engine_id = "017576662512468239146:omuauf_lfve"  # Using default Google CSE ID

    if not api_key:
        st.write("Error: GOOGLE_SEARCH_API_KEY not found in environment variables")
        return None

    # Clean company name and create query
    company_name = company_name.strip().replace("&", "and")  # Replace & with 'and'
    query = company_name  # Just use the company name

    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": search_engine_id,
        "num": 5
    }

    try:
        st.write(f"Searching for: {company_name}")
        st.write(f"Using query: {query}")

        response = requests.get(url, params=params)
        st.write(f"Response status: {response.status_code}")

        if response.status_code != 200:
            st.write("Error response:")
            st.write(response.text)  # Show the error message
            return None

        results = response.json()

        if "items" in results:
            st.write(f"Found {len(results['items'])} results")
            for item in results["items"]:
                url = item.get("link", "")
                title = item.get("title", "")
                st.write(f"Checking result: {title} - {url}")

                # Skip common non-company websites
                if not any(site in url.lower() for site in [
                    'linkedin.com', 'facebook.com', 'twitter.com', 'instagram.com',
                    'youtube.com', 'bloomberg.com', 'crunchbase.com', 'google.com'
                ]):
                    st.write(f"Found website for {company_name}: {url}")
                    return url
        else:
            st.write("No items in response. Full response:")
            st.write(results)

        st.write(f"No suitable website found for {company_name}")
        return None

    except Exception as e:
        st.write(f"Error searching for company website: {str(e)}")
        st.write("Request URL:", url)
        st.write("Request params:", params)
        return None


def scraped_email_step(profiles):
    if "emails_scraped" not in st.session_state:
        st.session_state["emails_scraped"] = {}

    if st.button("1. Start Scraping Emails"):
        # First, group profiles by company
        companies = {}
        for profile in profiles:
            company = profile["Company"]
            if company not in companies:
                companies[company] = []
            companies[company].append(profile)

        # Then process each company
        with st.spinner("Processing companies..."):
            for company, company_profiles in companies.items():
                if company not in st.session_state["emails_scraped"]:
                    st.markdown(f"##### Processing {company}")

                    # First, search for LinkedIn profiles
                    st.write(f"Finding profiles at {company}...")
                    all_profiles = []
                    for profile in company_profiles:
                        query = f"{profile['Name']} {company} site:linkedin.com/in/"
                        found_profiles = linkedin_search_pse(query, company, max_results=1)
                        all_profiles.extend(found_profiles)

                    if all_profiles:
                        # Then search for company website
                        st.write("Finding company website...")
                        company_url = company_website_search(company)

                        if company_url:
                            st.write(f"Scanning website: {company_url}")
                            emails = scrape_emails_with_playwright(company_url, company, max_emails=3)

                            if emails:
                                st.session_state["emails_scraped"][company] = emails
                                st.markdown(f"- Found {len(emails)} email(s)")
                                for email in emails:
                                    st.write(f"  • {email}")
                            else:
                                st.markdown("- No valid emails found")
                        else:
                            st.markdown("- Could not find company website")
                    else:
                        st.markdown("- Could not verify LinkedIn profiles")


def scrape_emails_with_playwright(url, company_name, max_emails=10):
    """Scrape emails from a website using requests and BeautifulSoup"""
    try:
        # Add http:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Use requests with headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all text content
        text_content = soup.get_text()

        # Also look for mailto links
        mailto_links = [a.get('href') for a in soup.find_all('a', href=True) if 'mailto:' in a.get('href', '')]
        mailto_emails = [link.replace('mailto:', '').strip() for link in mailto_links]

        # Regular expression for finding emails in text
        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
        text_emails = re.findall(email_pattern, text_content)

        # Combine both sources of emails
        all_emails = text_emails + mailto_emails

        # Clean and validate emails
        valid_emails = []
        for email in all_emails:
            # Clean the email
            email = email.strip().lower()
            # Basic validation
            if re.match(email_pattern, email) and '.' in email.split('@')[1]:
                valid_emails.append(email)

        # Remove duplicates and limit results
        unique_emails = list(set(valid_emails))[:max_emails]

        # Debug information
        st.write(f"Debug: Found {len(text_emails)} emails in text")
        st.write(f"Debug: Found {len(mailto_links)} mailto links")
        st.write(f"Debug: Total unique valid emails: {len(unique_emails)}")

        return unique_emails

    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return []


def linkedin_search_for_top_companies(roles, companies, city, max_results_per_company=3):
    """Search for LinkedIn profiles at specific companies"""
    all_profiles = []

    for company in companies:
        for role in roles:
            # Create search query
            query = f"{role} {company} {city} site:linkedin.com/in/"

            try:
                profiles = linkedin_search_pse(query, company, max_results=max_results_per_company)
                if profiles:
                    all_profiles.extend(profiles)
            except Exception as e:
                st.error(f"Error searching for {role} at {company}: {str(e)}")
                continue

    return all_profiles


def generate_emails_threaded(df, user_name, email_type, email_topic, max_workers=5):
    """Generate emails in parallel using ThreadPoolExecutor"""

    def generate_single_email(row):
        try:
            prompt = f"""
            Write a {email_type} email:
            From: {user_name}
            To: {row['name']} ({row['job title']})
            Topic: {email_topic}

            Make it professional but friendly. Keep it concise.
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional email writer. Write in British English."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )

            return response.choices[0].message["content"].strip()
        except Exception as e:
            st.error(f"Error generating email for {row['name']}: {str(e)}")
            return f"Error generating email: {str(e)}"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        emails = list(executor.map(generate_single_email, [row for _, row in df.iterrows()]))

    return emails


def setup_gmail_smtp():
    st.subheader("📧 Setup Gmail SMTP")

    # Clear SMTP settings button
    if 'smtp_email' in st.session_state:
        if st.button("Clear SMTP Settings"):
            for key in ['smtp_email', 'smtp_password', 'smtp_email_input', 'smtp_password_input']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    st.markdown("""
        ### How to get your App Password:
        1. Go to your Google Account settings
        2. Enable 2-Step Verification if not already enabled
        3. Go to Security → App passwords
        4. Select 'Mail' and 'Other' as device
        5. Copy the 16-character password

        ⚠️ Never use your main Gmail password here. Always use an App Password!
    """)

    # Create form
    with st.form("smtp_setup_form", clear_on_submit=False):
        smtp_email = st.text_input(
            "Gmail Address:",
            help="Enter your Gmail address"
        )
        smtp_password = st.text_input(
            "App Password:",
            type="password",
            help="Enter the 16-character App Password from Google"
        )
        submitted = st.form_submit_button("Save SMTP Settings")

        if submitted:
            st.write("Form submitted!")  # Debug info

            if not smtp_email or not smtp_password:
                st.error("Both email and password are required!")
                return False

            if not smtp_email.endswith('@gmail.com'):
                st.error("Please enter a valid Gmail address")
                return False

            try:
                st.info("Attempting to connect to Gmail...")
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()

                st.info("Attempting login...")
                server.login(smtp_email, smtp_password)

                st.info("Closing connection...")
                server.quit()

                # Save to session state
                st.session_state.smtp_email = smtp_email
                st.session_state.smtp_password = smtp_password

                st.success("✅ SMTP settings saved successfully!")
                st.balloons()
                return True

            except smtplib.SMTPAuthenticationError:
                st.error("Authentication failed. Please check your email and App Password.")
                return False
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
                return False

    # Show current status
    if 'smtp_email' in st.session_state:
        st.success(f"Currently configured email: {st.session_state.smtp_email}")

        # Test connection button
        if st.button("Test Connection"):
            try:
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(st.session_state.smtp_email, st.session_state.smtp_password)
                server.quit()
                st.success("✅ Connection test successful!")
            except Exception as e:
                st.error(f"Connection test failed: {str(e)}")

    return 'smtp_email' in st.session_state


def send_email(to_email, subject, body):
    if 'smtp_email' not in st.session_state or 'smtp_password' not in st.session_state:
        st.error("SMTP settings not configured!")
        return False

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = st.session_state.smtp_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Create SMTP session
        st.info("Connecting to Gmail...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        st.info("Logging in...")
        server.login(st.session_state.smtp_email, st.session_state.smtp_password)

        st.info("Sending email...")
        server.send_message(msg)
        server.quit()

        st.success("✅ Email sent successfully!")
        return True

    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False


# Then load environment variables
load_dotenv()

# Access API keys securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_KEY = os.getenv("MAPS_KEY")
SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
CSE_ID_LINKEDIN = os.getenv("GOOGLE_CSE_ID_LINKEDIN")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# App title and configuration
st.set_page_config(page_title="Odjo AI CRM", layout="wide")

# Get the absolute path to the static folder
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
logo_path = os.path.join(static_dir, "logo.png")

# Create a container with fixed width for logo and title
container = st.container()
with container:
    cols = st.columns([1, 6, 20])  # Adjust these numbers to control spacing
    with cols[0]:
        st.markdown("""
            <style>
                [data-testid="stImage"] {
                    min-width: 90px !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        if os.path.exists(logo_path):
            st.image(logo_path, width=90, caption=None)
        else:
            st.error(f"Logo not found. Please ensure it exists in the static folder.")
    with cols[1]:
        st.title("")
    with cols[2]:
        st.write("")

# Custom styling
st.markdown("""
    <style>
        .css-1d391kg {border-radius: 10px; padding: 20px;}
        div[data-testid="stHorizontalBlock"] {
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)

# Initialise session state for user info
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login page
if not st.session_state.logged_in:
    # Center the welcome message
    st.markdown("<h1 style='text-align: center;'>Welcome to Odjo AI</h1>", unsafe_allow_html=True)

    # Create some vertical space
    st.markdown("<br>", unsafe_allow_html=True)

    # Center the login form using columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        user_name = st.text_input("Your Name")
        user_email = st.text_input("Your Email")

        if st.button("Log In", use_container_width=True):
            if user_name and user_email:
                st.session_state.user_name = user_name
                st.session_state.user_email = user_email
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Please enter both name and email")

else:
    # Your existing navigation code here
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'menu'

    # Initialise tool_choice
    tool_choice = None

    # Sidebar navigation
    st.sidebar.title(f"Welcome, {st.session_state.user_name}!")

    # Initialise session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'menu'

    # Initialise tool_choice
    tool_choice = None

    # Sidebar navigation
    st.sidebar.title("Navigation")

    if st.sidebar.button('🏠 Menu'):
        st.session_state.current_page = 'menu'
        st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("📊 CRM Dashboard"):
        st.session_state.tool_choice = "CRM Dashboard"
        st.session_state.current_page = 'tool'
        st.rerun()

    if st.sidebar.button("🌐 Web Scraper"):
        st.session_state.tool_choice = "Web Scraper (Local Businesses)"
        st.session_state.current_page = 'tool'
        st.rerun()

    if st.sidebar.button("👥 LinkedIn Finder"):
        st.session_state.tool_choice = "LinkedIn Profile Finder"
        st.session_state.current_page = 'tool'
        st.rerun()

    if st.sidebar.button("📧 Email Generator"):
        st.session_state.tool_choice = "CSV Email Generator"
        st.session_state.current_page = 'tool'
        st.rerun()

    if st.sidebar.button("🔍 URL Scraper"):
        st.session_state.tool_choice = "URL Scraper"
        st.session_state.current_page = 'tool'
        st.rerun()

    # Main navigation menu
    if st.session_state.current_page == 'menu':
        st.title("🚀 Welcome to Odjo AI")

        col1, col2 = st.columns(2)
        col3, col4, col5 = st.columns(3)

        with col1:
            if st.button("📊 CRM Dashboard", use_container_width=True):
                st.session_state.tool_choice = "CRM Dashboard"
                st.session_state.current_page = 'tool'
                st.rerun()

        with col2:
            if st.button("🌐 Web Scraper", use_container_width=True):
                st.session_state.tool_choice = "Web Scraper (Local Businesses)"
                st.session_state.current_page = 'tool'
                st.rerun()

        with col3:
            if st.button("👥 LinkedIn Finder", use_container_width=True):
                st.session_state.tool_choice = "LinkedIn Profile Finder"
                st.session_state.current_page = 'tool'
                st.rerun()

        with col4:
            if st.button("📧 Email Generator", use_container_width=True):
                st.session_state.tool_choice = "CSV Email Generator"
                st.session_state.current_page = 'tool'
                st.rerun()

        with col5:
            if st.button("🔍 URL Scraper", use_container_width=True):
                st.session_state.tool_choice = "URL Scraper"
                st.session_state.current_page = 'tool'
                st.rerun()

    # Footer
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
    st.sidebar.markdown("Made with ❤️ by Odjo AI")

    # Set tool_choice from session state when not on menu
    if st.session_state.current_page == 'tool':
        tool_choice = st.session_state.tool_choice

    # ALL YOUR EXISTING TOOL CODE GOES HERE
    # (CRM Dashboard, Web Scraper, LinkedIn Finder, etc.)

    # Initialise session state for leads
    if "leads" not in st.session_state:
        st.session_state["leads"] = pd.DataFrame(
            columns=["Name", "Email", "Status", "Notes", "Last Contacted", "Contact Method"])

    # Initialise session state for steps
    if "step" not in st.session_state:
        st.session_state["step"] = 1


    # Function to move forward in the steps
    def go_forward():
        if st.session_state["step"] < 6:  # Adjust for the number of steps you have
            st.session_state["step"] += 1


    # Function to move back in the steps
    def go_back():
        if st.session_state["step"] > 1:
            st.session_state["step"] -= 1


    # Initialise other session state variables
    if "icps" not in st.session_state:
        st.session_state["icps"] = []

    if "businesses" not in st.session_state:
        st.session_state["businesses"] = []

    if "contacts" not in st.session_state:
        st.session_state["contacts"] = []

    if "outreach_context" not in st.session_state:
        st.session_state["outreach_context"] = ""

    # Main Streamlit UI
    st.markdown("#### Odjo AI - Welcome to Your New Sales Team")

    # Initialise session state
    if "step" not in st.session_state:
        st.session_state["step"] = 1

    total_steps = 6
    progress = st.progress(min(st.session_state["step"] / total_steps, 1.0))

    # Web Scraper Tool
    if tool_choice == "Web Scraper (Local Businesses)":
        # Step 1: Collect Name and Location
        if st.session_state["step"] == 1:
            st.header("Step 1: Tell Us About You")
            st.session_state["username"] = st.text_input("What is your name?", st.session_state.get("username", ""))
            st.session_state["location"] = st.text_input("Where are you located? (Town or Postcode)",
                                                         st.session_state.get("location", ""))
            if st.button("Next", key="proceed_step1"):
                if st.session_state["username"] and st.session_state["location"]:
                    st.session_state["step"] = 2
                    st.rerun()
                else:
                    st.warning("Please enter your name and location to proceed.")

        # Step 2: Define Service and Audience
        elif st.session_state["step"] == 2:
            st.header(f"Step 2: Define Your Service and Audience, {st.session_state['username']}")

            st.session_state["service"] = st.text_input(
                "Describe your service: (e.g., Graphic Designer specializing in logos)",
                st.session_state.get("service", "")
            )
            st.session_state["audience"] = st.text_input(
                "Who is your target audience?",
                st.session_state.get("audience", "")
            )
            st.session_state["description"] = st.text_area(
                "Optional: Provide a detailed description of what you do:",
                st.session_state.get("description", "")
            )

            if st.session_state["description"]:
                detailed_keywords = extract_keywords_with_gpt(st.session_state["description"])
                st.markdown(f"**Keywords based on your description:** {', '.join(detailed_keywords)}")

            if st.session_state["service"]:
                extracted_keywords = extract_keywords_with_gpt(st.session_state["service"])
                st.markdown(f"**Keywords based on your service:** {', '.join(extracted_keywords)}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back", key="back_step2"):
                    st.session_state["step"] = 1
                    st.rerun()
            with col2:
                if st.button("Next", key="proceed_step2"):
                    if st.session_state["service"] and st.session_state["audience"]:
                        st.session_state["step"] = 3
                        st.rerun()
                    else:
                        st.warning("Please fill in all fields to proceed.")

        # Step 3: Choose Distance and Find Businesses
        elif st.session_state["step"] == 3:
            st.header("Step 3: Choose Your Distance for Freelancer Market")

            distance = st.slider("Select maximum distance for searching businesses (in km):", 1, 100, 10)
            st.session_state["distance"] = distance

            location = st.session_state.get("location", "the location")

            # Add map visualization
            if location:
                try:
                    geolocator = GoogleV3(api_key=MAPS_KEY)
                    location_obj = geolocator.geocode(location)
                    if location_obj:
                        latitude = location_obj.latitude
                        longitude = location_obj.longitude
                        zoom_level = 12 if distance <= 10 else 10 if distance <= 30 else 8 if distance <= 50 else 6

                        m = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
                        folium.Circle(
                            location=[latitude, longitude],
                            radius=distance * 1000,
                            color="blue",
                            fill=True,
                            fill_opacity=0.2,
                        ).add_to(m)

                        st.write(f"Map showing businesses within {distance} km of {location}:")
                        st.components.v1.html(m._repr_html_(), height=500)
                except Exception as e:
                    st.error(f"Error displaying map: {str(e)}")

            if st.button("🔍 Search Businesses", key="find_businesses_step3"):
                if st.session_state.get("location"):
                    with st.spinner("Searching for businesses..."):
                        businesses = find_businesses_with_websites(
                            st.session_state["location"],
                            st.session_state.get("audience", ""),
                            distance * 1000
                        )
                        st.session_state["businesses"] = businesses

                        business_websites = [business["website"] for business in businesses if "website" in business]
                        contacts = scrape_contacts_from_websites(business_websites)
                        st.session_state["contacts"] = contacts

                        if businesses:
                            st.success(f"Found {len(businesses)} businesses with websites.")
                            st.subheader("Businesses Found:")
                            for idx, business in enumerate(businesses, start=1):
                                emails = contacts.get(business["website"], ["No email found"])
                                email_list = ", ".join(emails)
                                st.markdown(f"""
                                    **{idx}. {business['name']}**
                                    - Address: {business['address']}
                                    - Website: [Visit Website]({business['website']})
                                    - Email(s): {email_list}
                                """)
                        else:
                            st.warning(
                                "No businesses found in the specified area. Try adjusting the distance or location.")
                else:
                    st.warning("Please provide a location in Step 1.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back", key="back_step3"):
                    st.session_state["step"] = 2
                    st.rerun()
            with col2:
                if st.button("Next", key="proceed_step3"):
                    if st.session_state.get("contacts"):
                        st.session_state["step"] = 4
                        st.rerun()
                    else:
                        st.warning("Please search for businesses first.")

        # Step 4: Generate Emails for Contacts
        elif st.session_state["step"] == 4:
            st.header("Step 4: Generate Emails for Contacts")

            raw_contacts = st.session_state.get("contacts", {})
            clean_contacts = []
            for website, emails in raw_contacts.items():
                if isinstance(emails, list):
                    for email in emails:
                        if isinstance(email, str) and "@" in email:
                            clean_contacts.append({"website": website, "email": email})

            num_emails = len(clean_contacts)
            st.markdown(f"**Found {num_emails} email address{'es' if num_emails != 1 else ''}.**")

            email_type = st.radio("Select Email Type", ["Initial", "Follow up", "Final"])
            tone_of_voice = st.selectbox(
                "Select Tone of Voice",
                ["Friendly", "Professional", "Persuasive", "Formal", "Funny", "Yoda", "Arabic", "Poem"]
            )

            if st.button("Generate Emails"):
                if not clean_contacts:
                    st.warning("No valid emails found. Generating a generic email template.")
                    generic_email = generate_email_with_gpt(
                        template_type=email_type,
                        username=st.session_state.get("username", ""),
                        service=st.session_state.get("service", ""),
                        audience=st.session_state.get("audience", ""),
                        contact_name="[Contact Name]",
                        context=f"Tone: {tone_of_voice}",
                        description=st.session_state.get("description", "")
                    )
                    st.text_area("Generic Email Template:", generic_email, height=200)
                else:
                    with st.spinner("Generating personalised emails..."):
                        generated_emails = []
                        for contact in clean_contacts[:5]:  # Limit to 5 emails
                            email_content = generate_email_with_gpt(
                                template_type=email_type,
                                username=st.session_state.get("username", ""),
                                service=st.session_state.get("service", ""),
                                audience=st.session_state.get("audience", ""),
                                contact_name=contact["email"],
                                context=f"Tone: {tone_of_voice}",
                                description=st.session_state.get("description", "")
                            )
                            generated_emails.append({
                                "email": contact["email"],
                                "website": contact["website"],
                                "content": email_content
                            })
                        st.session_state["generated_emails"] = generated_emails

                        for idx, email in enumerate(generated_emails, 1):
                            st.markdown(f"### Email {idx} - To: {email['email']}")
                            st.text_area(f"Content {idx}:", email['content'], height=200)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back", key="back_step4"):
                    st.session_state["step"] = 3
                    st.rerun()
            with col2:
                if st.button("Next", key="proceed_step4"):
                    st.session_state["step"] = 5
                    st.rerun()

        # Step 5: Explore Demand for Your Service
        elif st.session_state["step"] == 5:
            st.header("Step 5: Explore Demand for Your Service")

            service_input = st.text_input("Enter a keyword or phrase related to your service:",
                                          value=st.session_state.get("service", ""))
            num_results = st.slider("Number of results to fetch:", 1, 10, 5)

            if st.button("Search Demand"):
                with st.spinner("Analyzing market demand..."):
                    # Search Mumsnet
                    mumsnet_results = google_custom_search(
                        f'site:mumsnet.com "{service_input}"',
                        "AIzaSyAsxu182FL1faktMPQvzw-OhXSp9lyM4tw",
                        "86de12f13a0b6428d",
                        num_results
                    )

                    # Search Facebook with multiple phrases
                    facebook_results = []
                    search_phrases = [
                        f'site:facebook.com "Looking for a {service_input}"',
                        f'site:facebook.com "Need a {service_input}"',
                        f'site:facebook.com "Searching for a {service_input}"',
                        f'site:facebook.com "Can anyone recommend a {service_input}"',
                        f'site:facebook.com "Looking to hire a {service_input}"'
                    ]

                    for phrase in search_phrases:
                        results = google_custom_search(
                            phrase,
                            "AIzaSyAsxu182FL1faktMPQvzw-OhXSp9lyM4tw",
                            "86de12f13a0b6428d",
                            num_results
                        )
                        facebook_results.extend(results)

                    # Remove duplicates from Facebook results
                    seen_links = set()
                    unique_facebook_results = []
                    for result in facebook_results:
                        if result['link'] not in seen_links:
                            seen_links.add(result['link'])
                            unique_facebook_results.append(result)

                    if mumsnet_results or unique_facebook_results:
                        st.success("Found relevant discussions!")

                        if mumsnet_results:
                            st.subheader("Mumsnet Discussions")
                            for result in mumsnet_results:
                                st.markdown(f"- [{result['title']}]({result['link']})")

                        if unique_facebook_results:
                            st.subheader("Facebook Posts")
                            for result in unique_facebook_results:
                                st.markdown(f"- [{result['title']}]({result['link']})")
                    else:
                        st.warning("No relevant discussions found. Try different keywords.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back", key="back_step5"):
                    st.session_state["step"] = 4
                    st.rerun()
            with col2:
                if st.button("Next", key="proceed_step5"):
                    st.session_state["step"] = 6
                    st.rerun()

        # Step 6: Thank You for Using Odjo AI
        elif st.session_state["step"] == 6:
            st.header("Thank You for Using Odjo AI! 🎉")

            st.markdown("""
                ## Here's what you've accomplished today:
                - Explored potential customers for your service.
                - Generated personalised outreach emails.
                - Researched demand for your services in the UK across platforms like Mumsnet and Facebook.
            """)

            st.markdown("""
                ### Next Steps:
                - Follow up on the contacts you've identified.
                - Tailor your services based on the demand you've researched.
                - Continue to use Odjo AI to grow your business!
            """)

            st.success("We're excited to see your company's journey thrive! 🚀")

            st.markdown("---")
            st.markdown(
                "If you have any feedback or suggestions, we'd love to hear from you. Contact us at [support@odjo.co.uk](mailto:support@odjo.co.uk).")

            # Navigation buttons in three columns
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Back", key="back_step6"):
                    st.session_state["step"] = 5
                    st.rerun()
            with col2:
                if st.button("Restart", key="restart_app"):
                    # Reset session state to restart the tool
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.session_state["step"] = 1
                    st.rerun()
            with col3:
                st.markdown("[Visit Odjo.co.uk](https://odjo.co.uk)")

    # LinkedIn Profile Finder Tool
    elif tool_choice == "LinkedIn Profile Finder":
        st.markdown("#### Odjo AI LinkedIn Scraper")

        # Step 1: User Input
        input_text = st.text_input("Target audience:")
        city = st.text_input("City (optional):")

        # Radio button to choose search type
        search_type = st.radio("Search type:", ["Top companies in industry", "Specific companies"])

        # Show specific companies input only if that option is selected
        specific_companies = None
        if search_type == "Specific companies":
            specific_companies = st.text_input("Enter company names (separate by comma):")

        if st.button("1. Analyze"):
            response = process_user_input(input_text)
            if response:
                roles, titles, industry = extract_roles_and_industry(response)
                st.markdown("##### Results:")
                st.markdown(f"- Roles: {', '.join(roles)}")
                st.markdown(f"- Industry: {industry}")
                st.session_state.update({
                    "roles": roles + titles,
                    "industry": industry,
                    "city": city,
                    "search_type": search_type,
                    "specific_companies": [comp.strip() for comp in
                                           specific_companies.split(",")] if specific_companies else []
                })
            else:
                st.error("Analysis failed")

        if st.button("2. Search"):
            if "roles" not in st.session_state or not st.session_state.get("roles"):
                st.warning("Please analyze first")
            else:
                roles = st.session_state.get("roles", [])
                city = st.session_state.get("city", None)

                # Use different search approach based on search type
                if st.session_state.get("search_type") == "Specific companies":
                    if st.session_state.get("specific_companies"):
                        companies = st.session_state["specific_companies"]
                        with st.spinner(f"Searching profiles in {', '.join(companies)}..."):
                            linked_profiles = linkedin_search_for_top_companies(roles, companies, city,
                                                                                max_results_per_company=3)
                    else:
                        st.warning("Please enter company names")
                        linked_profiles = []
                else:
                    industry = st.session_state.get("industry", "")
                    with st.spinner("Finding top companies..."):
                        companies = get_top_companies(industry, max_companies=2)
                    with st.spinner("Searching profiles..."):
                        linked_profiles = linkedin_search_for_top_companies(roles, companies, city,
                                                                            max_results_per_company=3)

                st.session_state["linked_profiles"] = linked_profiles
                if linked_profiles:
                    st.markdown(f"##### Found {len(linked_profiles)} Profiles:")
                    for profile in linked_profiles:
                        st.markdown(f"""
                        - **{profile.get('Name', 'N/A')}** | {profile.get('Title', 'N/A')}  
                          {profile.get('Company', 'N/A')} | [Profile]({profile.get('Source', '#')})
                        """)
                else:
                    st.warning("No profiles found")

        # Step 3: Email Generation
        if "linked_profiles" in st.session_state and st.session_state["linked_profiles"]:
            st.markdown("##### 3. Email Generation")
            profiles = st.session_state.get("linked_profiles", [])
            scraped_email_step(profiles)

    # CSV Email Generator Tool
    elif tool_choice == "CSV Email Generator":
        st.title("CSV Email Generator")
        st.write("Upload your CSV file, and we'll generate personalised emails for each contact!")

        user_name = st.text_input("Your Name:")
        email_type = st.selectbox("Email Type:", ["Sales Pitch", "Follow Up", "Newsletter"])
        email_topic = st.text_area("What is the email about?")

        uploaded_file = st.file_uploader("Upload CSV (Name, Email, Job Title required)", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            required_cols = ['name', 'email', 'job title']

            if not all(col.lower() in df.columns for col in required_cols):
                st.error("CSV must contain: Name, Email, Job Title")
            else:
                st.success("File validated successfully!")
                st.write("Preview:", df.head())

                if st.button("Generate Emails"):
                    with st.spinner("Generating emails..."):
                        emails = generate_emails_threaded(df, user_name, email_type, email_topic)
                        df['generated_email'] = emails

                        st.success("Emails generated!")
                        st.write("Preview of generated emails:")
                        st.dataframe(df[['name', 'email', 'generated_email']])

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Results",
                            csv,
                            "generated_emails.csv",
                            "text/csv"
                        )


    # Then add the new URL Scraper tool section
    elif tool_choice == "URL Scraper":
        st.title("URL Email Scraper")
        st.write("Enter a URL to find email addresses on that webpage.")

        # URL input
        url = st.text_input("Enter URL:", placeholder="https://example.com")

        if st.button("Scan URL"):
            if url:
                with st.spinner("Scanning webpage for emails..."):
                    try:
                        emails = scrape_emails_with_playwright(url, "Custom URL", max_emails=10)

                        if emails:
                            st.success(f"Found {len(emails)} email(s)!")

                            # Display results in a nice format
                            st.markdown("##### Found Emails:")
                            for email in emails:
                                st.markdown(f"- `{email}`")

                            # Add download button
                            if len(emails) > 0:
                                df = pd.DataFrame(emails, columns=["Email"])
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download Emails as CSV",
                                    csv,
                                    "emails.csv",
                                    "text/csv",
                                    key='download-csv'
                                )
                        else:
                            st.warning("No email addresses found on this webpage.")

                    except Exception as e:
                        st.error(f"Error scanning URL: {str(e)}")
            else:
                st.warning("Please enter a URL to scan.")

    elif tool_choice == "CRM Dashboard":
        st.title("📊 CRM Dashboard")

        # Create a grid of buttons for CRM features
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("👥 Contact Management", use_container_width=True):
                st.session_state.crm_feature = "Contact Management"
                st.rerun()

        with col2:
            if st.button("📧 Email Generation", use_container_width=True):
                st.session_state.crm_feature = "Email Generation"
                st.rerun()

        with col3:
            if st.button("📈 Lead Tracking", use_container_width=True):
                st.session_state.crm_feature = "Lead Tracking"
                st.rerun()

        # Second row of buttons
        col4, col5 = st.columns(2)

        with col4:
            if st.button("🔄 Campaign Sequences", use_container_width=True):
                st.session_state.crm_feature = "Campaign Sequences"
                st.rerun()

        with col5:
            if st.button("🤖 AI Assistant", use_container_width=True):
                st.session_state.crm_feature = "AI Assistant"
                st.rerun()

        # Add some spacing
        st.markdown("---")

        # Initialise session state for contacts if not exists
        if 'contacts_df' not in st.session_state:
            st.session_state.contacts_df = pd.DataFrame({
                'name': [
                    # Muslim names
                    'Ullah Samer',
                    'Fatima Khan',
                    'Omar Abdullah',

                    # Jewish names
                    'David Cohen',
                    'Sarah Goldstein',
                    'Rachel Levy',

                    # Chinese names
                    'Li Wei',
                    'Zhang Min',
                    'Wang Hui',

                    # Christian names
                    'Michael O\'Connor',
                    'Maria Rodriguez',
                    'John Smith',

                    # African/Black names
                    'Kwame Mensah',
                    'Zainab Okoro',
                    'Marcus Johnson'
                ],
                'email': [
                    'samerullah@hotmail.co.uk',
                    'fatima.k@example.com',
                    'omar.a@example.com',
                    'david.c@example.com',
                    'sarah.g@example.com',
                    'rachel.l@example.com',
                    'li.wei@example.com',
                    'zhang.min@example.com',
                    'wang.hui@example.com',
                    'michael.o@example.com',
                    'maria.r@example.com',
                    'john.s@example.com',
                    'kwame.m@example.com',
                    'zainab.o@example.com',
                    'marcus.j@example.com'
                ],
                'status': [
                    'Lead', 'Customer', 'Prospect',
                    'Customer', 'Lead', 'Prospect',
                    'Lead', 'Customer', 'Prospect',
                    'Lead', 'Customer', 'Prospect',
                    'Customer', 'Lead', 'Prospect'
                ],
                'last_contact': [
                    '2024-03-01', '2024-03-05', '2024-03-10',
                    '2024-03-02', '2024-03-06', '2024-03-11',
                    '2024-03-03', '2024-03-07', '2024-03-12',
                    '2024-03-04', '2024-03-08', '2024-03-13',
                    '2024-03-05', '2024-03-09', '2024-03-14'
                ],
                'notes': [
                    'Interested in service', 'Active customer', 'Follow up needed',
                    'Regular buyer', 'New lead', 'Requested demo',
                    'Meeting scheduled', 'Premium customer', 'Sent proposal',
                    'Needs follow-up', 'Long-term client', 'Initial contact',
                    'High potential', 'Recent inquiry', 'In negotiations'
                ]
            })

        # Display the selected feature's content
        if "crm_feature" in st.session_state:
            if st.session_state.crm_feature == "Contact Management":
                st.header("👥 Contact Management")

                # CSV Upload
                uploaded_file = st.file_uploader("Upload Contacts CSV", type=['csv'])
                if uploaded_file:
                    st.session_state.contacts_df = pd.read_csv(uploaded_file)
                    st.success("Contacts uploaded successfully!")

                # Display and Edit Contacts
                st.dataframe(st.session_state.contacts_df, use_container_width=True)

                # Add New Contact Form
                st.subheader("Add New Contact")
                col1, col2 = st.columns(2)
                with col1:
                    new_name = st.text_input("Name")
                    new_email = st.text_input("Email")
                with col2:
                    new_status = st.selectbox("Status", ["Lead", "Prospect", "Customer"])
                    new_notes = st.text_area("Notes")

                if st.button("Add Contact"):
                    new_contact = pd.DataFrame({
                        'name': [new_name],
                        'email': [new_email],
                        'status': [new_status],
                        'last_contact': [pd.Timestamp.now().strftime('%Y-%m-%d')],
                        'notes': [new_notes]
                    })
                    st.session_state.contacts_df = pd.concat([st.session_state.contacts_df, new_contact],
                                                             ignore_index=True)
                    st.success("Contact added successfully!")

            elif st.session_state.crm_feature == "Email Generation":
                st.header("📧 Email Generation")

                # Email generation form
                with st.form("email_generation_form"):
                    col1, col2 = st.columns(2)

                    with col1:
                        contact = st.selectbox("Select Contact", st.session_state.contacts_df['name'])
                        email_type = st.selectbox("Email Type", ["Introduction", "Follow-up", "Proposal", "Thank You"])

                    with col2:
                        custom_subject = st.text_input("Custom Subject (optional)")
                        additional_notes = st.text_area("Additional Notes (optional)",
                                                        help="Add any specific points you want to include in the email")

                    generate_button = st.form_submit_button("Generate Email")

                    if generate_button:
                        with st.spinner("Generating email..."):
                            try:
                                # Get contact data
                                contact_data = st.session_state.contacts_df[
                                    st.session_state.contacts_df['name'] == contact
                                    ].iloc[0]

                                # Create prompt for GPT
                                prompt = f"""
                                Write a professional {email_type.lower()} email:
                                From: {st.session_state.user_name}
                                To: {contact} ({contact_data['job_title'] if 'job_title' in contact_data else ''})
                                Type: {email_type}
                                Additional Notes: {additional_notes}

                                Make it professional but friendly. Keep it concise.
                                """

                                # Generate email content
                                response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are a professional email writer."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.7
                                )

                                generated_email = response.choices[0].message["content"].strip()

                                # Store in session state
                                st.session_state.current_email = {
                                    "to": contact_data['email'] if 'email' in contact_data else '',
                                    "subject": custom_subject or f"{email_type} - {st.session_state.user_name}",
                                    "body": generated_email
                                }

                                # Show success message
                                st.success("✉️ Email generated successfully!")

                            except Exception as e:
                                st.error(f"Error generating email: {str(e)}")

                # Show generated email if available
                if "current_email" in st.session_state:
                    st.markdown("### Generated Email")

                    # Email preview
                    with st.expander("📧 Preview Email", expanded=True):
                        st.text(f"From: {st.session_state.user_email}")
                        st.text("To: " + st.session_state.current_email["to"])
                        st.text("Subject: " + st.session_state.current_email["subject"])
                        st.markdown("---")
                        st.write(st.session_state.current_email["body"])

                    # Edit options
                    with st.expander("✏️ Edit Email"):
                        edited_subject = st.text_input("Edit Subject",
                                                       value=st.session_state.current_email["subject"])
                        edited_body = st.text_area("Edit Body",
                                                   value=st.session_state.current_email["body"],
                                                   height=300)

                        if st.button("Update Email"):
                            st.session_state.current_email["subject"] = edited_subject
                            st.session_state.current_email["body"] = edited_body
                            st.success("✅ Email updated!")

                        # Demo send button
                        if st.button("📤 Send Email"):
                            st.success(f"✉️ Email sent successfully from {st.session_state.user_email}!")
                            # Update contact's last_contact date
                            mask = st.session_state.contacts_df['name'] == contact
                            st.session_state.contacts_df.loc[mask, 'last_contact'] = pd.Timestamp.now().strftime(
                                '%Y-%m-%d')

            elif st.session_state.crm_feature == "Lead Tracking":
                st.header("📈 Lead Tracking")

                # Sample metrics with clickable elements
                metrics = {
                    'Total Leads': {'value': 150, 'trend': '+12% this month'},
                    'Open Rate': {'value': '45%', 'trend': '-3% vs last month'},
                    'Reply Rate': {'value': '22%', 'trend': '+5% vs last month'},
                    'Conversion Rate': {'value': '8%', 'trend': '+1% vs last month'}
                }

                # Display metrics as clickable elements
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Leads", metrics['Total Leads']['value'], metrics['Total Leads']['trend'])
                    if st.button("📊 Analyze Leads"):
                        with st.spinner("Analyzing lead data..."):
                            prompt = """Analyze this lead data and provide actionable insights:
                            - Total Leads: 150 (+12% this month)
                            - Current sources: Direct search, Referrals, Social Media
                            - Industry breakdown: Tech (40%), Services (35%), Retail (25%)

                            Provide 3-4 specific, actionable recommendations to improve lead generation.
                            Format with bullet points and prioritise by impact."""

                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a CRM and sales analytics expert."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.7
                            )

                            with st.expander("💡 Lead Generation Insights", expanded=True):
                                st.markdown(response.choices[0].message["content"])

                with col2:
                    st.metric("Open Rate", metrics['Open Rate']['value'], metrics['Open Rate']['trend'])
                    if st.button("📧 Improve Open Rates"):
                        with st.spinner("Analyzing email performance..."):
                            prompt = """Analyze this email performance data and provide specific improvements:
                            - Current open rate: 45% (-3% vs last month)
                            - Current subject line average length: 8 words
                            - Most common send time: 10am local time
                            - Device breakdown: Mobile (65%), Desktop (35%)

                            Provide 3-4 specific, actionable recommendations to improve email open rates.
                            Include example subject lines and optimal timing."""

                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are an email marketing optimization expert."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.7
                            )

                            with st.expander("💡 Email Open Rate Insights", expanded=True):
                                st.markdown(response.choices[0].message["content"])

                with col3:
                    st.metric("Reply Rate", metrics['Reply Rate']['value'], metrics['Reply Rate']['trend'])
                    if st.button("💬 Boost Replies"):
                        with st.spinner("Analyzing reply patterns..."):
                            prompt = """Analyze this email engagement data and provide improvements:
                            - Current reply rate: 22% (+5% vs last month)
                            - Average email length: 180 words
                            - Call-to-action placement: Bottom of email
                            - Most effective CTAs: Questions, Time-sensitive offers

                            Provide 3-4 specific, actionable recommendations to improve reply rates.
                            Include example CTAs and email structure tips."""

                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are an email engagement optimization expert."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.7
                            )

                            with st.expander("💡 Reply Rate Insights", expanded=True):
                                st.markdown(response.choices[0].message["content"])

                with col4:
                    st.metric("Conversion Rate", metrics['Conversion Rate']['value'],
                              metrics['Conversion Rate']['trend'])
                    if st.button("🎯 Optimise Conversions"):
                        with st.spinner("Analyzing conversion patterns..."):
                            prompt = """Analyze this conversion data and provide strategic improvements:
                            - Current conversion rate: 8% (+1% vs last month)
                            - Average sales cycle: 14 days
                            - Top converting channels: Direct (45%), Email (35%), Social (20%)
                            - Common objections: Price, Implementation time, Competition

                            Provide 3-4 specific, actionable recommendations to improve conversion rates.
                            Include objection handling techniques and follow-up strategies."""

                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a sales conversion optimization expert."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.7
                            )

                            with st.expander("💡 Conversion Rate Insights", expanded=True):
                                st.markdown(response.choices[0].message["content"])

                # Lead funnel visualization
                funnel_data = pd.DataFrame({
                    'Stage': ['Leads', 'Qualified', 'Meeting', 'Proposal', 'Won'],
                    'Count': [150, 80, 40, 20, 12]
                })

                fig = px.funnel(funnel_data, x='Count', y='Stage')
                st.plotly_chart(fig, use_container_width=True)

                # Lead status breakdown
                status_data = st.session_state.contacts_df['status'].value_counts()
                fig2 = px.pie(values=status_data.values, names=status_data.index, title='Lead Status Distribution')
                st.plotly_chart(fig2, use_container_width=True)

            elif st.session_state.crm_feature == "Campaign Sequences":
                st.header("🔄 Campaign Sequences")

                # Initialize all necessary session state variables
                if 'selected_contacts' not in st.session_state:
                    st.session_state.selected_contacts = []
                if 'campaign_type' not in st.session_state:
                    st.session_state.campaign_type = None
                if 'campaign_length' not in st.session_state:
                    st.session_state.campaign_length = 7
                if 'num_emails' not in st.session_state:
                    st.session_state.num_emails = 3
                if 'working_days_only' not in st.session_state:
                    st.session_state.working_days_only = True
                if 'email_contents' not in st.session_state:
                    st.session_state.email_contents = {}
                if 'industry' not in st.session_state:
                    st.session_state.industry = ""

                # Create tabs
                setup_tab, contacts_tab, content_tab, review_tab = st.tabs([
                    "📋 Campaign Setup",
                    "👥 Select Contacts",
                    "✍️ Content Strategy",
                    "🚀 Review & Launch"
                ])

                with setup_tab:
                    st.subheader("Campaign Configuration")

                    # Campaign Type Selection
                    campaign_types = {
                        "welcome_series": "👋 Welcome Series",
                        "lead_nurture": "🌱 Lead Nurture",
                        "product_launch": "🚀 Product Launch",
                        "follow_up": "🔄 Follow Up",
                        "re_engagement": "♻️ Re-engagement"
                    }

                    col1, col2 = st.columns(2)
                    with col1:
                        selected_type = st.selectbox(
                            "Campaign Type",
                            options=list(campaign_types.keys()),
                            format_func=lambda x: campaign_types[x],
                            key="campaign_type_select"
                        )
                        st.session_state.campaign_type = selected_type

                    with col2:
                        st.session_state.industry = st.text_input(
                            "Target Industry",
                            value=st.session_state.industry,
                            placeholder="e.g., Technology, Healthcare"
                        )

                    # Campaign Timeline Settings
                    st.markdown("### Campaign Timeline")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.session_state.num_emails = st.number_input(
                            "Number of Emails",
                            min_value=1,
                            max_value=10,
                            value=st.session_state.num_emails
                        )

                    with col2:
                        st.session_state.working_days_only = st.checkbox(
                            "Working Days Only",
                            value=st.session_state.working_days_only
                        )

                    st.session_state.campaign_length = st.slider(
                        "Campaign Duration (Days)",
                        min_value=1,
                        max_value=30,
                        value=st.session_state.campaign_length
                    )

                with contacts_tab:
                    st.subheader("Select Campaign Contacts")

                    # Search and filter
                    search = st.text_input("🔍 Search Contacts", "")

                    # Filter contacts based on search
                    filtered_contacts = st.session_state.contacts_df[
                        st.session_state.contacts_df['name'].str.contains(search, case=False) |
                        st.session_state.contacts_df['email'].str.contains(search, case=False)
                        ]

                    # Display contacts with select all option
                    select_all = st.checkbox("Select All Contacts")

                    if select_all:
                        st.session_state.selected_contacts = filtered_contacts['name'].tolist()
                    else:
                        st.session_state.selected_contacts = st.multiselect(
                            "Select Contacts for Campaign",
                            options=filtered_contacts['name'].tolist(),
                            default=st.session_state.selected_contacts
                        )

                    # Show selected contacts count and list
                    st.markdown(f"""
                        <div style='padding: 10px; background-color: #1E1E1E; border-radius: 5px; margin: 10px 0;'>
                            <p style='color: #FFFFFF; margin: 0;'>✅ Selected {len(st.session_state.selected_contacts)} contacts</p>
                        </div>
                    """, unsafe_allow_html=True)

                    if st.session_state.selected_contacts:
                        st.markdown("### Selected Contacts")
                        for contact in st.session_state.selected_contacts:
                            contact_data = filtered_contacts[filtered_contacts['name'] == contact].iloc[0]
                            st.markdown(f"""
                                <div style='padding: 10px; background-color: #2D2D2D; border-radius: 5px; margin: 5px 0;'>
                                    <p style='color: #FFFFFF; margin: 0;'>
                                        👤 {contact}<br>
                                        📧 {contact_data['email']}<br>
                                        📊 Status: {contact_data['status']}
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)

                with content_tab:
                    st.markdown("### Email Sequence Configuration")

                    for i in range(st.session_state.num_emails):
                        with st.expander(f"Email {i + 1} Configuration", expanded=i == 0):
                            col1, col2 = st.columns(2)
                            with col1:
                                trigger = st.selectbox(
                                    f"Trigger for Email {i + 1}",
                                    options=[
                                        "Campaign Start" if i == 0 else "Days After Previous",
                                        "No Open",
                                        "Opened but No Reply",
                                        "Clicked but No Action"
                                    ],
                                    key=f"trigger_{i}"
                                )

                            with col2:
                                if i > 0:
                                    wait_days = st.number_input(
                                        "Days to Wait",
                                        min_value=1,
                                        max_value=5,
                                        value=2,
                                        key=f"wait_{i}"
                                    )

                            if st.button(f"Generate Email {i + 1} Content", key=f"gen_email_{i}"):
                                with st.spinner("Crafting email..."):
                                    email_prompt = f"""
                                    Write a concise, professional {st.session_state.campaign_type.replace('_', ' ')} email:
                                    Position: Email {i + 1} of {st.session_state.num_emails}
                                    Industry: {st.session_state.industry}
                                    Trigger: {trigger}

                                    Requirements:
                                    - Keep it under 100 words
                                    - Be direct and value-focused
                                    - Include one relevant industry trend
                                    - Use psychology principles for higher engagement
                                    - End with a clear call to action
                                    """

                                    response = openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system",
                                             "content": "You are an expert in sales email copywriting."},
                                            {"role": "user", "content": email_prompt}
                                        ]
                                    )

                                    email_content = response.choices[0].message["content"].strip()
                                    st.session_state.email_contents[f'email_{i}'] = email_content

                                    st.markdown(f"""
                                        <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid #333;'>
                                            <p style='color: #FFFFFF; margin: 0;'><strong>Generated Email Content:</strong></p>
                                            <p style='color: #FFFFFF; margin-top: 10px;'>{email_content.replace(chr(10), '<br>')}</p>
                                        </div>

                                        <div style='background-color: #2D2D2D; padding: 10px; border-radius: 5px; margin-top: 10px; border: 1px solid #333;'>
                                            <p style='color: #FFFFFF; margin: 0;'>📋 Trigger: {trigger}</p>
                                            {'<p style="color: #FFFFFF; margin: 0;">⏰ Wait time: ' + str(wait_days) + ' days</p>' if i > 0 else ''}
                                            <p style='color: #FFFFFF; margin: 0;'>📝 Email {i + 1} of {st.session_state.num_emails}</p>
                                        </div>
                                    """, unsafe_allow_html=True)

                            elif f'email_{i}' in st.session_state.email_contents:
                                st.markdown(f"""
                                    <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid #333;'>
                                        <p style='color: #FFFFFF; margin: 0;'><strong>Current Email Content:</strong></p>
                                        <p style='color: #FFFFFF; margin-top: 10px;'>{st.session_state.email_contents[f'email_{i}'].replace(chr(10), '<br>')}</p>
                                    </div>
                                """, unsafe_allow_html=True)

                with review_tab:
                    st.subheader("Campaign Review & Launch")

                    st.markdown("""
                        <div style='padding: 20px; background-color: #1E1E1E; border-radius: 10px;'>
                            <h3 style='color: #FFFFFF;'>Campaign Summary</h3>
                        </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("📧 Total Emails:", st.session_state.num_emails)
                        st.write("👥 Selected Contacts:", len(st.session_state.selected_contacts))
                        st.write("📅 Duration:", f"{st.session_state.campaign_length} days")
                    with col2:
                        st.write("⚙️ Working Days Only:", st.session_state.working_days_only)
                        st.write("🎯 Campaign Type:", st.session_state.campaign_type.replace('_', ' ').title())
                        st.write("🏢 Industry:", st.session_state.industry or "Not specified")

                    # Launch button and countdown
                    if st.button("🚀 Launch Campaign", type="primary"):
                        if len(st.session_state.selected_contacts) == 0:
                            st.error("Please select at least one contact before launching the campaign.")
                        elif len(st.session_state.email_contents) < st.session_state.num_emails:
                            st.error("Please generate content for all emails before launching.")
                        else:
                            countdown = 5
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for i in range(countdown, -1, -1):
                                progress_bar.progress((countdown - i) / countdown)
                                status_text.markdown(f"### 🚀 Campaign starting in {i}...")
                                time.sleep(1)

                            status_text.markdown("### ✅ Campaign Successfully Launched!")
                            st.balloons()

                            # Update contact status
                            for contact in st.session_state.selected_contacts:
                                mask = st.session_state.contacts_df['name'] == contact
                                st.session_state.contacts_df.loc[mask, 'campaign_status'] = 'Active'


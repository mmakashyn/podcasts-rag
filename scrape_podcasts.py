import requests
from bs4 import BeautifulSoup
import re
import os

def clean_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def scrape_transcript(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    transcript_body = soup.find('div', class_='podcast-episode-transcript__body')
    title_element = soup.find('h1', class_='podcast-episode-header__title')
    episode_title = title_element.text.strip() if title_element else "Untitled Episode"
    
    if not transcript_body:
        print(f"Transcript not found for: {episode_title}")
        return None, episode_title
    
    paragraphs = transcript_body.find_all('p')
    
    transcript = []
    for p in paragraphs:
        parts = [part.strip() for part in p.stripped_strings]
        if parts:
            first_part = parts[0].strip('"')
            transcript.append(first_part)
            transcript.extend(parts[1:])
        transcript.append('')
    
    return '\n'.join(transcript), episode_title

def save_transcript(transcript, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(transcript)
    print(f"Transcript saved to {filename}")

def scrape_podcast_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    buttons = soup.find_all('a', class_='btn btn-primary btn--outline podcast-default-btn')
    links = [button['href'] for button in buttons if 'href' in button.attrs]
    return links

# Create the resources_scrapped directory if it doesn't exist
os.makedirs('resources_scrapped', exist_ok=True)

base_url = "https://courses.floodlightgrp.com/podcasts/head-heart-boots?items=100&page="
total_episodes = 0

for page in range(1, 14):
    url = f"{base_url}{page}"
    print(f"Scraping page {page}: {url}")
    
    podcast_links = scrape_podcast_links(url)
    
    for link in podcast_links:
        print(f"  Scraping transcript from: {link}")
        transcript, episode_title = scrape_transcript(link)
        
        if transcript:
            clean_title = clean_filename(episode_title)
            filename = f"resources_scrapped/{clean_title}.txt"
            save_transcript(transcript, filename)
            total_episodes += 1
    
    print(f"Finished scraping page {page}")
    print()

print(f"Total episodes scraped: {total_episodes}")
print("Scraping completed.")
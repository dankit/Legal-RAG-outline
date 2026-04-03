import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import time


class IowaWebScraper:
    """Scraper for downloading PDF documents from the Iowa Administrative Code website."""

    def __init__(self, base_url, download_folder, processed_folder):
        self.base_url = base_url
        self.download_folder = download_folder
        self.processed_folder = processed_folder

    def create_output_folder(self):
        """Create the output folder if it doesn't exist."""
        if not os.path.exists(self.download_folder):
            os.makedirs(self.download_folder)

    def get_filename_from_url(self, url):
        """Extract filename from URL or generate one if not available."""
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
    
        if not filename or not filename.endswith('.pdf'):
            path_parts = parsed_url.path.strip('/').split('/')
            if path_parts:
                filename = path_parts[-1] + '.pdf'
            else:
                filename = 'document.pdf'
        
        return filename

    def download_pdf(self, url, filename):
        """Download a PDF file from the given URL."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
        
            file_path = os.path.join(self.download_folder, filename)
        
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            return False

    def find_pdf_links(self, soup, base_url):
        """Find all PDF links in the HTML content."""
        pdf_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href.lower().endswith('.pdf') or "analysis" in href.lower():
                continue
            filename = self.get_filename_from_url(href)
            if os.path.exists(os.path.join(self.download_folder, filename)) or os.path.exists(os.path.join(self.processed_folder, filename)):
                continue
            absolute_url = urljoin(base_url, href)
            pdf_links.append(absolute_url)
        
        return pdf_links

    def scrape_administrative_code(self):
        """Main function to scrape PDFs from the Iowa Administrative Code website."""
        self.create_output_folder()
        
        try:
            response = requests.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_links = self.find_pdf_links(soup, self.base_url)
            unique_pdf_links = list(dict.fromkeys(pdf_links))
            
            successful_downloads = 0
            for i, pdf_url in enumerate(unique_pdf_links, 1):
                filename = self.get_filename_from_url(pdf_url)
                
                if os.path.exists(os.path.join(self.download_folder, filename)) or \
                   os.path.exists(os.path.join(self.processed_folder, filename)):
                    continue
                
                print(f"[{i}/{len(unique_pdf_links)}] Downloading: {filename}")
                
                if self.download_pdf(pdf_url, filename):
                    successful_downloads += 1
                
                time.sleep(1)
                
            print(f"Downloaded {successful_downloads}/{len(unique_pdf_links)} PDFs")
        except Exception as e:
            print(f"Error during scraping: {str(e)}")

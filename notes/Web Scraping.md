
# open_source 
## https://github.com/scrapy/scrapy

## common crawl
https://commoncrawl.org/access-the-data/

# zenrows
https://app.zenrows.com/register
https://www.zenrows.com/blog/web-crawler-python#what-is-a-web-crawler-in-python

# using chrome


On a Linux machine, the process is similar, but there are a few adjustments you'll need to make. Here's a step-by-step guide:

### 1. Install Chrome:

You'll need to have Google Chrome installed. If it's not already, you can install it using:

For Debian/Ubuntu:

```bash
sudo apt update
sudo apt install google-chrome-stable
```

For CentOS/RedHat:

```bash
sudo yum install google-chrome-stable
```

### 2. Install ChromeDriver:

You'll also need the ChromeDriver which matches your Chrome version. You can usually find the appropriate version on the [ChromeDriver download page](https://sites.google.com/a/chromium.org/chromedriver/downloads).

Download and extract it:

```bash
wget https://chromedriver.storage.googleapis.com/<version>/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/bin/chromedriver
sudo chown root:root /usr/bin/chromedriver
sudo chmod +x /usr/bin/chromedriver
```

Replace `<version>` with the correct version.

### 3. Set up Selenium:

You'll need to install the required Python packages:

```bash
pip install selenium
```

### 4. Python Script:

You can now use Selenium just like in the example provided earlier, but with the path to the `chromedriver` you just installed:

```python
from selenium import webdriver

# Set up the driver
browser = webdriver.Chrome(executable_path='/usr/bin/chromedriver')

# Navigate to the page
browser.get("https://www.nasdaq.com/market-activity/stocks/googl/earnings")

# Grab content (as an example)
content = browser.page_source

# Always close the browser when done
browser.close()

# Print or process the content
print(content)
```

### 5. Optional: Run Headless:

If you're on a server without a GUI, you'll want to run Chrome in "headless" mode, which means it runs without displaying any GUI. You can do this by adjusting the WebDriver options:

```python
from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Set up the driver with the headless option
browser = webdriver.Chrome(executable_path='/usr/bin/chromedriver', options=options)

# ... rest of the code ...
```

Remember to always ensure you have the right permissions to scrape a website and respect their terms of service.
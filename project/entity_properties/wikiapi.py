import wikipediaapi as wapi

'''
1. query wikipedia for topic name => https://pypi.org/project/Wikipedia-API/
2. filter important text

:param entity: string to search article for
:return: text of article
'''
def get_text(entity):
	wiki = wapi.Wikipedia(language='en', extract_format=wapi.ExtractFormat.WIKI)
	page = wiki.page(entity)
	title = page.title
	text = page.text
	return text

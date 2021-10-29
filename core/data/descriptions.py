import os
import re

import bs4
import fire
import tidy


PARSER = 'html5lib'  # 'html.parser'
HEADER_TAGS = ['h2', 'h3']
INPUT_HEADER_NAMES = ['Input', '入力']
CONSTRAINT_HEADER_NAMES = ['Constraints', '制約', '入力形式']
# 制約 = Constraint.
# 入力形式 = Input format.


def as_soup(text, soup=None):
  if soup is not None:
    return soup
  text = str(tidy.parseString(text, add_xml_decl=0, tidy_mark=0, wrap=0))
  return bs4.BeautifulSoup(text, PARSER)


def extract_section_content(header_name, text, soup=None):
  soup = as_soup(text, soup=soup)
  input_header = soup.find(HEADER_TAGS, text=re.compile(rf'^\s*{header_name}\s*$'))
  input_text = get_text_following_header(input_header)
  return input_text.strip()


def extract_input_description(text, soup=None):
  return extract_section_content('Input', text, soup=soup)


def extract_input_constraints(text, soup=None):
  return extract_section_content('Constraints', text, soup=soup)


def extract_input_information(text, soup=None):
  soup = as_soup(text, soup=soup)
  info = []
  for header_name in INPUT_HEADER_NAMES + CONSTRAINT_HEADER_NAMES:
    content = extract_section_content(header_name, text, soup=soup)
    if content:
      info.append((header_name, content))
  return '\n\n'.join(
      f'{header_name}:\n{content}'
      for header_name, content in info)


def get_text_following_header(header):
  next_node = header
  lines = []
  while next_node is not None:
    next_node = next_node.nextSibling
    if next_node is None:
      break
    elif isinstance(next_node, bs4.Tag) and next_node.name in HEADER_TAGS:
      break
    elif isinstance(next_node, bs4.NavigableString):
      text = next_node.strip()
    else:
      text = next_node.get_text().strip()
    lines.append(text)
  return '\n'.join(lines)


def get_all_input_descriptions():
  no_text = 0
  no_text_ids = []
  no_info = 0
  no_info_ids = []
  good = 0
  for i in range(4053):
    path = f'/mnt/project-codenet-storage/Project_CodeNet/problem_descriptions/p{i:05d}.html'
    if not os.path.exists(path):
      no_text += 1
      no_text_ids.append(i)
      continue
    with open(path, 'r') as f:
      text = f.read()
    info = extract_input_information(text)
    if not info:
      no_info += 1
      no_info_ids.append(i)
    else:
      good += 1
    print('Index:', i)
    print(info)
    print()
  print('No text', no_text)
  print('No info', no_info)
  print('Good:', good)
  print('No info ids', no_info_ids)
  print('No text ids', no_text_ids)

if __name__ == '__main__':
  fire.Fire()


import smtplib
import json

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

ADMIN_SERVER = 'sentimentanalysispro@gmail.com'
ADMIN_PASSWORD = 'godnb52020!'

# email constants
MAIL_SUBJECT = 'Twitter json Report'

REPORT_NAME = 'Twitter_api/tweets.json'

# The project collaborators and their emails
PROJECT_GROUP = {
    'Gidi': "gidemsky26@gmail.com",
    'Daniella The Sis': "Daniella.kirshenbaum@gmail.com",
    'Oriya Aharon The best': "oriya717@gmail.com",
    'Batel Cohen the princess': "batel.cohen100@gmail.com",
    'Natali the mommy care': "Balulunatalie@gmail.com"
}


def separate_debug_print_big(title):
    # long line separator for debug line console print
    print('\n------------------------------' + title + '------------------------------\n')


def separate_debug_print_small(title):
    # short line separator for debug line console print
    print('\n---------------' + title + '---------------')


def all_group_emails(key=None):
    """
    Gets the asked member email
    :param key: member name
    :return: the email of the person or of all the group
    """
    if key is None:
        return PROJECT_GROUP.values()
    else:
        return PROJECT_GROUP[key]


def get_json_list(src_json_file):
    """
    Converts json file with the following structure: (Dict) tweets : list
    :param src_json_file: json file with the correct structure
    :return: list of the tweets
    """
    with open(src_json_file, 'r', encoding="utf-8") as json_file:
        json_all_dict_data = json.load(json_file)
    return json_all_dict_data['tweets']


def create_json_dict_file(json_list, json_file_name):
    """
    Convert the list to correct json file: (Dict) tweets : list
    :param json_list: list od tweets
    :param json_file_name: the destination file name
    :return: json file
    """
    json_all_list = {'tweets': json_list}
    if not json_file_name.endswith('.json'):
        json_file_name = json_file_name + '.json'
    with open(json_file_name, 'w', encoding='UTF-8') as fp:
        json.dump(json_all_list, fp, ensure_ascii=False, indent=3)


def check_tweets_number(src_json_file):
    """
    checks the number of the tweets in the json file
    :param src_json_file:
    :return: number of tweets
    """
    return len(get_json_list(src_json_file))


def send_report_by_email(mail_subject="No subject", body_text=None, file_path=None):
    """
    The sending e-mail function using gmail account
    :param mail_subject: The subject of the email to be written
    :param body_text: the body of the email (if needed)
    :param file_path: in case we have file - sends the file path to be sent
    :return: -
    """

    msg = MIMEMultipart()
    msg['From'] = ADMIN_SERVER
    msg['Subject'] = mail_subject

    msg.attach(MIMEText('Hey,\nThis is an automated e-mail report system.\n' + body_text, 'plain'))

    # in case there is file to be sent - prepare the file to email format
    if file_path is not None:
        attachment = open(file_path, 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= " + file_path)
        msg.attach(part)

    # converts all the text and files to email format
    cur_mail = msg.as_string()

    # SMTP server configuration and settings
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    # log-in to the gmail account and sends the mail
    server.login(ADMIN_SERVER, ADMIN_PASSWORD)
    server.sendmail(ADMIN_SERVER, all_group_emails(), cur_mail)

    # close user and SMTP connection
    server.quit()

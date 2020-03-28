"""
The main google API translation Class.
Class abilities:
    It can translate string and list to any supported language
    Print all the supported languages
    Detects the text language

    The main our using language are:
    Hebrew is 'iw'
    English is 'en'
    Arabic is 'ar'
"""

import six
import Utils as util
from google.cloud import translate_v2 as translate


class GoogleTranslateAPI(object):

    def __init__(self):
        self.translate_client = translate.Client()

    def translate(self, source_text, to_language, print_result=False, model=None):
        """
        Translates text into the target language.
        :param source_text: list or strings of strings to translate.
        :param to_language: The language to translate results into.
        :param print_result: If we want to print the result
        :param model: 'base' or 'nmt'. By default it using NMT.
        :return: The translated to the destination language string
        """

        # Modifies the text to utf-8 in case it doesn't
        if isinstance(source_text, six.binary_type):
            source_text = source_text.decode('utf-8')

        # Translates the source text.
        translated_text = self.translate_client.translate(values=source_text, target_language=to_language, model=model)

        # for debug purpose
        if print_result:
            util.separate_debug_print("Start Translation Result")
            print(u'Text to translate: {}'.format(
                translated_text['input']))
            print(u'The translation: {}'.format(
                translated_text['translatedText']))
            print(u'Detected source language: {}'.format(
                translated_text['detectedSourceLanguage']))
            util.separate_debug_print("End Translation Result")
        return translated_text['translatedText']

    def supp_language(self, language=None):
        """
        Lists all available languages and localizes them to the target language.
        :param language: The language to show the supported languages
        :return:
        """
        all_supported_language = self.translate_client.get_languages(target_language=language)

        util.separate_debug_print("Start the supported languages")
        for cur_language in all_supported_language:
            print(u'{name} ({language})'.format(**cur_language))
        util.separate_debug_print("End the supported languages")

    def language_detection(self, source_text, print_result=False):
        """
        Checks what is the language of the input text with probability
        :param source_text: The source text we want to detect the language
        :param print_result: To print the result, for the debug reason
        :return: The text language
        """

        language_result = self.translate_client.detect_language(source_text)

        # for debug reason
        if print_result:
            print('Text: {}'.format(source_text))
            print('Confidence: {}'.format(language_result['confidence']))
            print('Language: {}'.format(language_result['language']))

        return language_result['language']


if __name__ == '__main__':
    test_text = "Hey, My name is Gidi. I like movies, sex, and Disco-Dancing. This is test for the translation"
    test_text2 = "היי, שוב בדיקה!"
    google = GoogleTranslateAPI()
    google.supp_language()

    result = google.translate(test_text, 'ru', True)
    print(result)

    result = google.translate(test_text, 'iw', True)
    print(result)

    result = google.translate(test_text, 'ar', True)
    print(result)

    google.supp_language('iw')

    result = google.translate(test_text2, 'ar', True)
    print(result)

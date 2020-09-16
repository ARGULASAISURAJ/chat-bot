# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List, Union
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted, SlotSet, EventType, ConversationPaused
from rasa_sdk.forms import FormAction
import datetime
import requests
import json
import csv
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

INTENT_DESCRIPTION_MAPPING_PATH = "intent_description_mapping.csv"
DB_USER = os.getenv("DB_USER")
DB_PWD = os.getenv("DB_PWD")

class GetName(Action):
    
    def name(self) -> Text:
        return "action_get_name"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        connection = psycopg2.connect(user = DB_USER, password = DB_PWD, host = "cloudsql-proxy", port = "5432", database = "chatbot_db")
        cursor = connection.cursor()
        cursor.execute("SELECT first_name FROM scores ORDER BY id DESC LIMIT 1;")
        name = cursor.fetchone()[0]
        if (connection):
            cursor.close()
            connection.close()
        return [SlotSet("name", name)]

class GetPanasScore(Action):

    def name(self) -> Text:
        return "action_get_panas_score"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        connection = psycopg2.connect(user = DB_USER, password = DB_PWD, host = "cloudsql-proxy", port = "5432", database = "chatbot_db")
        cursor = connection.cursor()
        cursor.execute("SELECT panas_score FROM scores ORDER BY id DESC LIMIT 1;")
        score = cursor.fetchone()[0]
        if (connection):
            cursor.close()
            connection.close()
        if score == 0:
            # dispatcher.utter_message("But your Panas score is negative!")
            return [SlotSet("panas_score", "0")]
        else:
            return [SlotSet("panas_score", "1")]
        

class ActionDefaultAskAffirmation(Action):
    """Asks for an affirmation of the intent if NLU threshold is not met."""

    def name(self) -> Text:
        return "action_default_ask_affirmation"

    def __init__(self) -> None:
    
        self.intent_mappings = pd.read_csv(INTENT_DESCRIPTION_MAPPING_PATH)
        self.intent_mappings.fillna("", inplace=True)
        self.intent_mappings.entities = self.intent_mappings.entities.map(
            lambda entities: {e.strip() for e in entities.split(",")}
        )

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:

        intent_ranking = tracker.latest_message.get("intent_ranking", [])
        if len(intent_ranking) > 1:
            diff_intent_confidence = intent_ranking[0].get(
                "confidence"
            ) - intent_ranking[1].get("confidence")
            if diff_intent_confidence < 0.2:
                intent_ranking = intent_ranking[:2]
            else:
                intent_ranking = intent_ranking[:1]

        # for the intent name used to retrieve the button title, we either use
        # the name of the name of the "main" intent, or if it's an intent that triggers
        # the response selector, we use the full retrieval intent name so that we
        # can distinguish between the different sub intents
        first_intent_names = [
            intent.get("name", "")
            for intent in intent_ranking
        ]

        message_title = (
            "Sorry, I'm not sure I've understood you correctly 🤔 Do you mean..."
        )

        entities = tracker.latest_message.get("entities", [])
        entities = {e["entity"]: e["value"] for e in entities}

        entities_json = json.dumps(entities)

        buttons = []
        for intent in first_intent_names:
            button_title = self.get_button_title(intent)
            if "/" in intent:
                # here we use the button title as the payload as well, because you
                # can't force a response selector sub intent, so we need NLU to parse
                # that correctly
                buttons.append({"title": button_title, "payload": button_title})
            else:
                buttons.append(
                    {"title": button_title, "payload": f"/{intent}{entities_json}"}
                )

        buttons.append({"title": "Something else", "payload": "/out_of_scope"})

        dispatcher.utter_message(text=message_title, buttons=buttons)

        return []

    def get_button_title(self, intent: Text) -> Text:
        utterance_query = self.intent_mappings.intent == intent

        utterances = self.intent_mappings[utterance_query].button.tolist()

        if len(utterances) > 0:
            button_title = utterances[0]
        else:
            utterances = self.intent_mappings[utterance_query].button.tolist()
            button_title = utterances[0] if len(utterances) > 0 else intent

        return button_title



class ActionSkipToActivity(Action):
    def name(self) -> Text:
        return "action_skip_to_activity"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(template="utter_skip_to_activity_fb")

        return [UserUtteranceReverted()]


class ActionSeverityScore(Action):
    def name(self) -> Text:
        return "action_severity_score"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        emotion_intensity = tracker.get_slot('emotion_intensity')
        emotion_bother = tracker.get_slot('emotion_bother')
        emotion_impact = tracker.get_slot('emotion_impact')
        score = emotion_intensity + emotion_bother + emotion_impact
        if score <= 6:
            dispatcher.utter_message(text="Even though this hasn't taken a huge toll on you, I'm sure it’s still hard to be dealing with these feelings. I'm really glad you shared this with me.")
            return [SlotSet("severity_score", "1")]
        elif 7 <= score <= 11:
            dispatcher.utter_message(text="That sounds really tough. I can see why it's got you feeling this way.")
            return [SlotSet("severity_score", "2")]
        elif score >= 12:
            dispatcher.utter_message(text="It sounds like this has been really upsetting and I can see why. The way you're feeling right now is totally valid.")
            return [SlotSet("severity_score", "3")]
        else:
            return []


class SeverityFormNoButtons(FormAction):
    
    def name(self):
        return "severity_form_no_buttons"

    @staticmethod
    def required_slots(tracker):
        return ["emotion_intensity", "emotion_bother", "emotion_impact"]

    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
        return {
            "emotion_intensity": [
                self.from_entity(entity="emotion_intensity")
            ],
            "emotion_bother": [
                self.from_entity(entity="emotion_bother")
            ],
            "emotion_impact": [
                self.from_entity(entity="emotion_impact")
            ]
        }

    def validate_emotion_intensity(self, value: Text, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if value.lower() in ["low", "moderate", "high"]:
            return {"emotion_intensity": value}
        else:
            return {"emotion_intensity": None}

    def validate_emotion_bother(self, value: Text, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if value.lower() in ["a little", "moderately", "a lot"]:
            return {"emotion_bother": value}
        else:
            return {"emotion_bother": None}

    def validate_emotion_impact(self, value: Text, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if value.lower() in ["low", "moderate", "high"]:
            return {"emotion_impact": value}
        else:
            return {"emotion_impact": None}

    def submit(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        if tracker.get_slot('emotion_intensity') == "low":
            emotion_intensity = 1
        elif tracker.get_slot('emotion_intensity') == "moderate":
            emotion_intensity = 2
        elif tracker.get_slot('emotion_intensity') == "high":
            emotion_intensity = 3
        if tracker.get_slot('emotion_bother') == "a little":
            emotion_bother = 1
        elif tracker.get_slot('emotion_bother') == "moderately":
            emotion_bother = 2
        elif tracker.get_slot('emotion_bother') == "a lot":
            emotion_bother = 3
        if tracker.get_slot('emotion_impact') == "low":
            emotion_impact = 1
        elif tracker.get_slot('emotion_impact') == "moderate":
            emotion_impact = 2
        elif tracker.get_slot('emotion_impact') == "high":
            emotion_impact = 3

        score = emotion_intensity + emotion_bother + emotion_impact
        if score <= 3:
            dispatcher.utter_message(template="utter_low_severity")
            return [SlotSet("severity_score", "1")]
        elif 4 <= score <= 6:
            dispatcher.utter_message(template="utter_moderate_severity")
            return [SlotSet("severity_score", "2")]
        elif score >= 7:
            dispatcher.utter_message(template="utter_high_severity")
            return [SlotSet("severity_score", "3")]
        else:
            return []


class SeverityFormWithButtons(FormAction):
    
    def name(self):
        return "severity_form_with_buttons"

    @staticmethod
    def required_slots(tracker):
        return ["emotion_intensity", "emotion_bother", "emotion_impact"]

    def submit(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        emotion_intensity = tracker.get_slot('emotion_intensity')
        emotion_bother = tracker.get_slot('emotion_bother')
        emotion_impact = tracker.get_slot('emotion_impact')

        score = emotion_intensity + emotion_bother + emotion_impact
        if score <= 3:
            dispatcher.utter_message(template="utter_low_severity")
            return [SlotSet("severity_score", "1")]
        elif 4 <= score <= 6:
            dispatcher.utter_message(template="utter_moderate_severity")
            return [SlotSet("severity_score", "2")]
        elif score >= 7:
            dispatcher.utter_message(template="utter_high_severity")
            return [SlotSet("severity_score", "3")]
        else:
            return []
from flask import Blueprint, jsonify, session, current_app
import os
from authlib.integrations.flask_client import OAuth

bp = Blueprint("auth", __name__)

def google_token():
    if current_app.debug:
        return {
            "email": "dummy@gmail.com",
        }
    
    if 'google_token' in session:
        me = oauth.google.get('userinfo')
        return jsonify({'data': me.data})
    return None

@bp.route('/login/oidc/google')
def login():
    return google.authorize(callback=url_for('authorized', _external=True))

def config_oauth(app):
    google_client_id = os.environ["GOOGLE_CLIENT_ID"]
    google_client_secret = os.environ["GOOGLE_CLIENT_SECRET"]
    google_redirect_uri = os.environ["GOOGLE_REDIRECT_URI"]

    oauth = OAuth()
    oauth.init_app(app)

    google = oauth.register(
        'google',
        consumer_key=google_client_id,
        consumer_secret=google_client_secret,
        request_token_params={
            'scope': 'email',
        },
        base_url='https://www.googleapis.com/oauth2/v1/',
        request_token_url=None,
        access_token_method='POST',
        access_token_url='https://accounts.google.com/o/oauth2/token',
        authorize_url='https://accounts.google.com/o/oauth2/auth',
    )

    app.register_blueprint(bp)


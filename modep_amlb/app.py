import logging
import argparse
import flask
import tornado.wsgi
import tornado.httpserver

from modep_common.models import AnonUser

from modep_amlb import app


# @celery.task()
# def add_together(a, b):
#     print('Adding', a, b)
#     return a + b


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    logging.info(("Tornado server starting on port {}".format(port)))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="enable debug mode", type=int, default=1)
    parser.add_argument(
        "-p", "--port", help="port to serve content on", type=int, default=5000
    )
    parser.add_argument(
        "--host", help="host to serve content on", type=str, default="0.0.0.0"
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    # test the DB connection
    with app.app_context():
        print(AnonUser.query.all())

    # from flask import current_app
    # with app.app_context():
    #     print('Trying to add')
    #     result = add_together.delay(23, 42)  # 65
    #     print('Add result', result.wait())

    if args.debug == 0:
        start_tornado(app, args.port)
    elif args.debug == 1:
        app.run(debug=True, host=args.host, port=args.port)
    else:
        raise Exception("unknown args.debug: %i" % args.debug)

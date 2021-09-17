import argparse
import logging
import flask
import tornado.wsgi
import tornado.httpserver

from modep_common.models import AnonUser, User
from modep_common import settings

from modep_amlb import app


def print_objs():
    """ Test DB connection by printing some objects """
    with app.app_context():
        print('-'*100)
        print('objects')
        print("AnonUser")
        print(AnonUser.query.all())
        print("User")
        print(User.query.all())


# @celery.task()
# def add_together(a, b):
#     print('Adding', a, b)
#     return a + b


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    logging.info(('Tornado server starting on port {}'.format(port)))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', help='enable debug mode',
                        type=int, default=1)
    parser.add_argument('-p', '--port', help='port to serve content on',
                        type=int, default=5000)
    parser.add_argument('--host', help='host to serve content on',
                        type=str, default='0.0.0.0')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    print_objs()

    # from flask import current_app

    # with app.app_context():
    #     print('Trying to add')
    #     result = add_together.delay(23, 42)  # 65
    #     print('Add result', result.wait())

    # with app.app_context():
    #     result = current_app.celery.send_task('tasks.update_framework', args=('c9afa411-6db2-448c-b74d-b4b3502f5b7d', '/tmp/tmp_ty5a3sl'))
    #     print(result.get())

    if args.debug == 0:
        start_tornado(app, args.port)
    elif args.debug == 1:
        app.run(debug=True, host=args.host, port=args.port)
    else:
        raise Exception('unknown args.debug: %i' % args.debug)

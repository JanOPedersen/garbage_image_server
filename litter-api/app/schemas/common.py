from marshmallow import Schema, fields


class TimingSchema(Schema):
    decode = fields.Float(required=True)
    preprocess = fields.Float(required=True)
    forward = fields.Float(required=True)
    postprocess = fields.Float(required=True)
    total = fields.Float(required=True)
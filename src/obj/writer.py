

def writer(prefix):

    def prefixed_writer(iterable):
        return prefix + " " + " ".join(str(x) for x in iterable)

    return prefixed_writer


write_vertex = writer('v')
write_face = writer('f')


def write_obj(name, obj):

    obj_string = "o " + name + "\n" + \
                 "\n".join(write_vertex(v) for v in obj['vertices']) + '\n' +\
                 "\n".join(write_face(f) for f in obj['faces']) + '\n'

    return obj_string

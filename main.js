'use strict';

var NODE_MODULE_PATH = 'node_modules';
var PROJECT_MODULE_PATH = 'libs';
var APP_PATH = '';

var requireConfig = {
    //except, if the module ID starts with "app",
    //load it from the js/app directory. paths
    //config is relative to the baseUrl, and
    //never includes a ".js" extension since
    //the paths config could be for a directory.
    paths: {
        'Vue': NODE_MODULE_PATH + '/vue/dist/vue.min',
        'vue': NODE_MODULE_PATH + '/requirejs-vue/requirejs-vue'
    },
    config: {

    },

    shim: {

    }

};

requirejs.config(requireConfig);

// Start the main app logic.
require(['Vue', 'vue'], function(Vue, vue){
    require(['vue!app'], function(theApp){
        theApp.$mount('#app');
    });
});

